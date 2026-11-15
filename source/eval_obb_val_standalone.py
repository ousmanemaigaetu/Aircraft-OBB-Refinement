#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Évaluateur OBB autonome pour le benchmark avions.
Aucune dépendance à MMDetection/MMCV/CUDA.

Entrées:
- annotations DOTA par image:
    x1 y1 x2 y2 x3 y3 x4 y4 plane difficult
- prédictions fusionnées:
    image_id score x1 y1 x2 y2 x3 y3 x4 y4

Métriques:
- AP50
- AP75
- mAP50-95 (IoU 0.50:0.05:0.95)
- interpolation AP à 101 points
- P / R / F1 à un seuil de score donné
- difficult ignorés
"""

from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import cv2


def norm_id(x: str) -> str:
    return Path(str(x)).stem


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.convexHull(a.astype(np.float32))
    b = cv2.convexHull(b.astype(np.float32))
    aa = abs(float(cv2.contourArea(a)))
    bb = abs(float(cv2.contourArea(b)))
    if aa <= 0 or bb <= 0:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
        inter = float(inter)
    except cv2.error:
        return 0.0
    union = aa + bb - inter
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def read_gt(gt_dir: Path, class_name="plane"):
    gt = {}
    n_useful = 0
    n_difficult = 0

    for fp in sorted(gt_dir.glob("*.txt")):
        items = []
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("imagesource") or s.startswith("gsd"):
                continue
            p = s.split()
            if len(p) < 9:
                continue
            try:
                coords = np.asarray(list(map(float, p[:8])), dtype=np.float32).reshape(4, 2)
            except Exception:
                continue
            if p[8] != class_name:
                continue

            difficult = 0
            if len(p) >= 10:
                try:
                    difficult = int(float(p[9]))
                except Exception:
                    difficult = 0

            items.append({"poly": coords, "difficult": difficult})
            if difficult:
                n_difficult += 1
            else:
                n_useful += 1
        gt[norm_id(fp.name)] = items

    if not gt:
        raise RuntimeError(f"Aucun GT trouvé dans {gt_dir}")
    return gt, n_useful, n_difficult


def read_preds(path: Path):
    preds = []
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        p = s.split()
        if len(p) < 10:
            bad += 1
            continue
        try:
            image_id = norm_id(p[0])
            score = float(p[1])
            poly = np.asarray(list(map(float, p[2:10])), dtype=np.float32).reshape(4, 2)
        except Exception:
            bad += 1
            continue
        preds.append({"image_id": image_id, "score": score, "poly": poly})

    preds.sort(key=lambda x: x["score"], reverse=True)
    if not preds:
        raise RuntimeError(f"Aucune prédiction trouvée dans {path}")
    return preds, bad


def ap101(rec, prec):
    recall_points = np.linspace(0.0, 1.0, 101)
    vals = []
    for r in recall_points:
        mask = rec >= r
        vals.append(float(np.max(prec[mask])) if np.any(mask) else 0.0)
    return float(np.mean(vals))


def evaluate_at_iou(gt, preds, n_useful, iou_thr, score_min=-math.inf):
    # état de matching indépendant pour chaque seuil IoU
    matched = {}
    for image_id, items in gt.items():
        matched[image_id] = np.zeros(len(items), dtype=bool)

    tp = []
    fp = []
    ignored = []

    for pred in preds:
        if pred["score"] < score_min:
            continue

        image_id = pred["image_id"]
        items = gt.get(image_id, [])

        if not items:
            tp.append(0)
            fp.append(1)
            ignored.append(False)
            continue

        ious = np.asarray([poly_iou(pred["poly"], g["poly"]) for g in items], dtype=np.float64)

        # 1) tenter d'apparier un GT utile non déjà pris
        useful_candidates = [
            i for i, g in enumerate(items)
            if not g["difficult"] and not matched[image_id][i]
        ]
        if useful_candidates:
            best_i = max(useful_candidates, key=lambda i: ious[i])
            best_iou = float(ious[best_i])
        else:
            best_i = None
            best_iou = -1.0

        if best_i is not None and best_iou >= iou_thr:
            matched[image_id][best_i] = True
            tp.append(1)
            fp.append(0)
            ignored.append(False)
            continue

        # 2) si la prédiction correspond à un difficult, elle est ignorée
        diff_candidates = [
            i for i, g in enumerate(items)
            if g["difficult"] and ious[i] >= iou_thr
        ]
        if diff_candidates:
            tp.append(0)
            fp.append(0)
            ignored.append(True)
            continue

        # 3) sinon FP
        tp.append(0)
        fp.append(1)
        ignored.append(False)

    tp = np.asarray(tp, dtype=np.int64)
    fp = np.asarray(fp, dtype=np.int64)
    ignored = np.asarray(ignored, dtype=bool)

    keep = ~ignored
    tp2 = tp[keep]
    fp2 = fp[keep]

    if len(tp2) == 0:
        return {
            "ap": 0.0,
            "rec": np.asarray([]),
            "prec": np.asarray([]),
            "tp": 0,
            "fp": 0,
            "fn": n_useful,
        }

    ctp = np.cumsum(tp2)
    cfp = np.cumsum(fp2)

    rec = ctp / max(n_useful, 1)
    prec = ctp / np.maximum(ctp + cfp, 1)

    ap = ap101(rec, prec)
    TP = int(ctp[-1])
    FP = int(cfp[-1])
    FN = int(max(n_useful - TP, 0))

    return {"ap": ap, "rec": rec, "prec": prec, "tp": TP, "fp": FP, "fn": FN}


def prf_at_score(gt, preds, n_useful, score_thr=0.50, iou_thr=0.50):
    out = evaluate_at_iou(gt, preds, n_useful, iou_thr=iou_thr, score_min=score_thr)
    tp, fp, fn = out["tp"], out["fp"], out["fn"]
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1, tp, fp, fn


def evaluate_model(name, gt, n_useful, preds, score_thr):
    iou_thresholds = np.arange(0.50, 0.951, 0.05)
    aps = {}

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    for thr in iou_thresholds:
        r = evaluate_at_iou(gt, preds, n_useful, float(thr))
        aps[round(float(thr), 2)] = r["ap"]
        print(f"AP@{thr:.2f}: {100*r['ap']:.4f} %")

    ap50 = aps[0.50]
    ap75 = aps[0.75]
    map5095 = float(np.mean(list(aps.values())))

    P, R, F1, TP, FP, FN = prf_at_score(
        gt, preds, n_useful, score_thr=score_thr, iou_thr=0.50
    )

    print()
    print(f"AP50       : {100*ap50:.4f} %")
    print(f"AP75       : {100*ap75:.4f} %")
    print(f"mAP50-95   : {100*map5095:.4f} %")
    print(f"Score >= {score_thr:.6f}, IoU >= 0.50")
    print(f"Precision  : {100*P:.4f} %")
    print(f"Recall     : {100*R:.4f} %")
    print(f"F1         : {100*F1:.4f} %")
    print(f"TP / FP / FN : {TP} / {FP} / {FN}")

    return {
        "ap50": ap50,
        "ap75": ap75,
        "map5095": map5095,
        "precision": P,
        "recall": R,
        "f1": F1,
        "tp": TP,
        "fp": FP,
        "fn": FN,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--s2-pred", type=Path, required=True)
    ap.add_argument("--yolo-pred", type=Path, required=True)
    ap.add_argument("--class-name", default="plane")
    ap.add_argument("--s2-score", type=float, default=0.2259,
                    help="Seuil F1 S2 choisi sur VAL. Défaut 0.2259")
    ap.add_argument("--yolo-score", type=float, default=0.566895,
                    help="Seuil F1 YOLO choisi sur VAL. Défaut 0.566895")
    args = ap.parse_args()

    # Protection simple : ce script est destiné à VAL pour cette étape
    for p in [args.gt_dir, args.s2_pred, args.yolo_pred]:
        parts = [x.lower() for x in Path(p).parts]
        if "test" in parts or any(x.startswith("test_") for x in parts):
            raise SystemExit(
                f"[STOP] Chemin TEST détecté: {p}\n"
                "Pour cette étape de recherche, utilise uniquement VAL."
            )

    gt, n_useful, n_difficult = read_gt(args.gt_dir, args.class_name)
    s2_preds, s2_bad = read_preds(args.s2_pred)
    y_preds, y_bad = read_preds(args.yolo_pred)

    print("ÉVALUATEUR OBB AUTONOME - VAL")
    print(f"Images annotées : {len(gt)}")
    print(f"GT utiles       : {n_useful}")
    print(f"GT difficult    : {n_difficult}")
    print(f"Prédictions S2  : {len(s2_preds)} (lignes invalides: {s2_bad})")
    print(f"Prédictions YOLO: {len(y_preds)} (lignes invalides: {y_bad})")

    s2 = evaluate_model("S²A-Net", gt, n_useful, s2_preds, args.s2_score)
    yo = evaluate_model("YOLO26l-OBB", gt, n_useful, y_preds, args.yolo_score)

    print()
    print("=" * 72)
    print("COMPARAISON YOLO - S²A-Net")
    print("=" * 72)
    print(f"Δ AP50     : {100*(yo['ap50']-s2['ap50']):+.4f} pts")
    print(f"Δ AP75     : {100*(yo['ap75']-s2['ap75']):+.4f} pts")
    print(f"Δ mAP50-95 : {100*(yo['map5095']-s2['map5095']):+.4f} pts")

    print()
    print("Référence attendue connue pour S²A-Net VAL epoch 11:")
    print("AP50 ≈ 94.171 %")
    print("AP75 ≈ 56.692 %")
    print("mAP50-95 ≈ 55.080 %")
    print()
    print("Si l'écart avec ces trois valeurs est > 0.1 point,")
    print("on arrête avant le diagnostic et on vérifie la convention d'évaluation.")


if __name__ == "__main__":
    main()
