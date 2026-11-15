#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic causal de localisation OBB sur VAL uniquement.

But:
1) Reproduire le matching standard à IoU >= 0.50 pour chaque modèle.
2) Garder les GT utiles détectés par LES DEUX modèles à IoU >= 0.50.
3) Sur exactement ces mêmes avions, comparer:
   - IoU exact
   - erreur de centre normalisée
   - erreur d'angle
   - erreur de longueur / largeur
   - effet de la taille et de l'orientation
   - relation score / IoU
   - qui franchit IoU 0.75

Aucune dépendance MMDetection/MMCV/CUDA.
Dépendances: numpy, opencv-python, matplotlib
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def norm_id(x):
    return Path(str(x)).stem


def poly_iou(a, b):
    a = cv2.convexHull(np.asarray(a, np.float32))
    b = cv2.convexHull(np.asarray(b, np.float32))
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


def rect_features(poly):
    pts = np.asarray(poly, np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)

    cx, cy = rect[0]
    edges = []
    for i in range(4):
        v = box[(i + 1) % 4] - box[i]
        L = float(np.linalg.norm(v))
        ang = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
        edges.append((L, ang))

    major, angle = max(edges, key=lambda z: z[0])
    minor = min(z[0] for z in edges)
    area = max(major * minor, 1e-12)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "major": float(major),
        "minor": float(minor),
        "area": float(area),
        "angle": float(angle),
        "aspect": float(major / max(minor, 1e-12)),
    }


def angle_error(a, b):
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


def read_gt(gt_dir, class_name="plane"):
    gt = {}
    n_useful = 0
    n_diff = 0

    for fp in sorted(Path(gt_dir).glob("*.txt")):
        arr = []
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("imagesource") or s.startswith("gsd"):
                continue
            p = s.split()
            if len(p) < 9 or p[8] != class_name:
                continue
            try:
                poly = np.asarray(list(map(float, p[:8])), np.float32).reshape(4, 2)
            except Exception:
                continue
            difficult = 0
            if len(p) >= 10:
                try:
                    difficult = int(float(p[9]))
                except Exception:
                    difficult = 0
            arr.append({"poly": poly, "difficult": difficult})
            if difficult:
                n_diff += 1
            else:
                n_useful += 1
        gt[norm_id(fp.name)] = arr

    if not gt:
        raise RuntimeError("Aucun GT trouvé.")
    return gt, n_useful, n_diff


def read_preds(path):
    by_img = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.strip().split()
        if len(p) < 10:
            continue
        try:
            image_id = norm_id(p[0])
            score = float(p[1])
            poly = np.asarray(list(map(float, p[2:10])), np.float32).reshape(4, 2)
        except Exception:
            continue
        by_img[image_id].append({"score": score, "poly": poly})

    for k in by_img:
        by_img[k].sort(key=lambda x: x["score"], reverse=True)
    return dict(by_img)


def standard_match_iou50(gt_items, pred_items, thr=0.50):
    """
    Matching type détection: prédictions triées par score décroissant,
    chaque GT utile ne peut être pris qu'une fois.
    Les difficult ne participent pas au diagnostic géométrique.
    Retour: gt_idx -> (pred, iou)
    """
    useful = [i for i, g in enumerate(gt_items) if not g["difficult"]]
    used = set()
    matches = {}

    for pred in pred_items:
        best_i = None
        best_iou = -1.0
        for i in useful:
            if i in used:
                continue
            v = poly_iou(pred["poly"], gt_items[i]["poly"])
            if v > best_iou:
                best_iou = v
                best_i = i
        if best_i is not None and best_iou >= thr:
            used.add(best_i)
            matches[best_i] = (pred, float(best_iou))
    return matches


def metrics_for_pair(gt_poly, pred_poly, iou, score):
    g = rect_features(gt_poly)
    p = rect_features(pred_poly)

    center_px = math.hypot(p["cx"] - g["cx"], p["cy"] - g["cy"])
    diag = math.hypot(g["major"], g["minor"])
    center_norm = center_px / max(diag, 1e-12)

    return {
        "score": float(score),
        "iou": float(iou),
        "center_err_norm": float(center_norm),
        "angle_err_deg": float(angle_error(p["angle"], g["angle"])),
        "major_abs_rel_err": float(abs(p["major"] - g["major"]) / max(g["major"], 1e-12)),
        "minor_abs_rel_err": float(abs(p["minor"] - g["minor"]) / max(g["minor"], 1e-12)),
        "area_ratio": float(p["area"] / max(g["area"], 1e-12)),
    }


def median(x):
    x = np.asarray(list(x), dtype=float)
    return float(np.median(x)) if len(x) else float("nan")


def mean(x):
    x = np.asarray(list(x), dtype=float)
    return float(np.mean(x)) if len(x) else float("nan")


def percentile(x, q):
    x = np.asarray(list(x), dtype=float)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def grouped_summary(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)

    out = []
    for group, rr in sorted(groups.items(), key=lambda z: str(z[0])):
        out.append({
            key: group,
            "n": len(rr),
            "s2_iou_mean": mean(r["s2_iou"] for r in rr),
            "yolo_iou_mean": mean(r["yolo_iou"] for r in rr),
            "delta_iou_mean": mean(r["delta_iou"] for r in rr),
            "s2_reach75_pct": 100.0 * mean(r["s2_iou"] >= 0.75 for r in rr),
            "yolo_reach75_pct": 100.0 * mean(r["yolo_iou"] >= 0.75 for r in rr),
            "s2_angle_err_med": median(r["s2_angle_err_deg"] for r in rr),
            "yolo_angle_err_med": median(r["yolo_angle_err_deg"] for r in rr),
            "s2_center_err_med": median(r["s2_center_err_norm"] for r in rr),
            "yolo_center_err_med": median(r["yolo_center_err_norm"] for r in rr),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--s2-pred", required=True, type=Path)
    ap.add_argument("--yolo-pred", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--class-name", default="plane")
    args = ap.parse_args()

    # Protection TEST
    for p in [args.gt_dir, args.s2_pred, args.yolo_pred, args.out_dir]:
        parts = [x.lower() for x in Path(p).parts]
        if "test" in parts or any(x.startswith("test_") for x in parts):
            raise SystemExit(f"[STOP] Chemin TEST détecté: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt, n_useful, n_diff = read_gt(args.gt_dir, args.class_name)
    s2pred = read_preds(args.s2_pred)
    ypred = read_preds(args.yolo_pred)

    # Terciles de taille sur TOUS les GT utiles de VAL
    areas = []
    for items in gt.values():
        for g in items:
            if not g["difficult"]:
                areas.append(rect_features(g["poly"])["area"])
    q33, q67 = np.quantile(np.asarray(areas), [1/3, 2/3])

    rows = []
    n_s2_tp50 = 0
    n_y_tp50 = 0

    for image_id, gt_items in gt.items():
        sm = standard_match_iou50(gt_items, s2pred.get(image_id, []), 0.50)
        ym = standard_match_iou50(gt_items, ypred.get(image_id, []), 0.50)

        n_s2_tp50 += len(sm)
        n_y_tp50 += len(ym)

        common = sorted(set(sm) & set(ym))
        for gi in common:
            gpoly = gt_items[gi]["poly"]
            gf = rect_features(gpoly)

            sp, siou = sm[gi]
            yp, yiou = ym[gi]

            smet = metrics_for_pair(gpoly, sp["poly"], siou, sp["score"])
            ymet = metrics_for_pair(gpoly, yp["poly"], yiou, yp["score"])

            if gf["area"] <= q33:
                size_bin = "small"
            elif gf["area"] <= q67:
                size_bin = "medium"
            else:
                size_bin = "large"

            a0 = int((gf["angle"] % 180.0) // 15) * 15
            angle_bin = f"{a0:03d}-{a0+15:03d}"

            row = {
                "image_id": image_id,
                "gt_idx": gi,
                "gt_area": gf["area"],
                "gt_major": gf["major"],
                "gt_minor": gf["minor"],
                "gt_aspect": gf["aspect"],
                "gt_angle_deg": gf["angle"],
                "size_bin": size_bin,
                "angle_bin": angle_bin,

                "s2_score": smet["score"],
                "s2_iou": smet["iou"],
                "s2_center_err_norm": smet["center_err_norm"],
                "s2_angle_err_deg": smet["angle_err_deg"],
                "s2_major_abs_rel_err": smet["major_abs_rel_err"],
                "s2_minor_abs_rel_err": smet["minor_abs_rel_err"],
                "s2_area_ratio": smet["area_ratio"],

                "yolo_score": ymet["score"],
                "yolo_iou": ymet["iou"],
                "yolo_center_err_norm": ymet["center_err_norm"],
                "yolo_angle_err_deg": ymet["angle_err_deg"],
                "yolo_major_abs_rel_err": ymet["major_abs_rel_err"],
                "yolo_minor_abs_rel_err": ymet["minor_abs_rel_err"],
                "yolo_area_ratio": ymet["area_ratio"],

                "delta_iou": ymet["iou"] - smet["iou"],
                "delta_angle_err": ymet["angle_err_deg"] - smet["angle_err_deg"],
                "delta_center_err": ymet["center_err_norm"] - smet["center_err_norm"],
                "delta_major_err": ymet["major_abs_rel_err"] - smet["major_abs_rel_err"],
                "delta_minor_err": ymet["minor_abs_rel_err"] - smet["minor_abs_rel_err"],
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("Aucun GT commun détecté par les deux modèles à IoU >= 0.50.")

    fields = list(rows[0].keys())
    write_csv(args.out_dir / "common_gt_iou50.csv", rows, fields)

    size_rows = grouped_summary(rows, "size_bin")
    angle_rows = grouped_summary(rows, "angle_bin")
    write_csv(args.out_dir / "summary_by_size.csv", size_rows, list(size_rows[0].keys()))
    write_csv(args.out_dir / "summary_by_angle.csv", angle_rows, list(angle_rows[0].keys()))

    # Catégories IoU75 sur les mêmes GT
    cat = Counter()
    for r in rows:
        s = r["s2_iou"] >= 0.75
        y = r["yolo_iou"] >= 0.75
        if s and y:
            cat["both_75"] += 1
        elif y and not s:
            cat["yolo_only_75"] += 1
        elif s and not y:
            cat["s2_only_75"] += 1
        else:
            cat["neither_75"] += 1

    # Corrélations simples pour identifier le facteur le plus associé au gain d'IoU
    diou = [r["delta_iou"] for r in rows]
    corr = {
        "angle": pearson(diou, [-r["delta_angle_err"] for r in rows]),
        "center": pearson(diou, [-r["delta_center_err"] for r in rows]),
        "major": pearson(diou, [-r["delta_major_err"] for r in rows]),
        "minor": pearson(diou, [-r["delta_minor_err"] for r in rows]),
    }

    # Résumé
    lines = []
    lines.append("DIAGNOSTIC CAUSAL OBB - VAL UNIQUEMENT")
    lines.append("=" * 72)
    lines.append(f"Images VAL               : {len(gt)}")
    lines.append(f"GT utiles                : {n_useful}")
    lines.append(f"GT difficult ignorés     : {n_diff}")
    lines.append(f"TP@0.50 S²A-Net          : {n_s2_tp50}")
    lines.append(f"TP@0.50 YOLO26l-OBB      : {n_y_tp50}")
    lines.append(f"GT communs TP@0.50       : {len(rows)}")
    lines.append("")
    lines.append("SUR LES MEMES AVIONS DETECTES PAR LES DEUX MODELES")
    lines.append("-" * 72)
    for model in ["s2", "yolo"]:
        lines.append(
            f"{model.upper():4s} IoU médian={median(r[model+'_iou'] for r in rows):.4f} | "
            f"angle err médiane={median(r[model+'_angle_err_deg'] for r in rows):.3f}° | "
            f"centre err médiane={median(r[model+'_center_err_norm'] for r in rows):.5f} | "
            f"longueur err médiane={100*median(r[model+'_major_abs_rel_err'] for r in rows):.2f}% | "
            f"largeur err médiane={100*median(r[model+'_minor_abs_rel_err'] for r in rows):.2f}%"
        )

    lines.append("")
    lines.append(f"ΔIoU médian YOLO-S2      : {median(diou):+.4f}")
    lines.append(f"ΔIoU moyen YOLO-S2       : {mean(diou):+.4f}")
    lines.append("")
    lines.append("PASSAGE DU SEUIL IoU=0.75 SUR LES GT COMMUNS")
    lines.append("-" * 72)
    for k in ["both_75", "yolo_only_75", "s2_only_75", "neither_75"]:
        n = cat[k]
        lines.append(f"{k:16s}: {n:4d} ({100*n/len(rows):6.2f} %)")

    lines.append("")
    lines.append("CORRELATION ENTRE GAIN D'IoU ET AMELIORATION GEOMETRIQUE")
    lines.append("(plus la valeur est positive, plus ce facteur accompagne le gain d'IoU)")
    lines.append("-" * 72)
    lines.append(f"Angle    : {corr['angle']:+.4f}")
    lines.append(f"Centre   : {corr['center']:+.4f}")
    lines.append(f"Longueur : {corr['major']:+.4f}")
    lines.append(f"Largeur  : {corr['minor']:+.4f}")
    lines.append("")
    lines.append(f"Terciles aire GT: q33={q33:.2f}, q67={q67:.2f}")
    lines.append("")
    lines.append("IMPORTANT: ce diagnostic est fait sur VAL. Aucun choix n'est fait sur TEST.")

    summary = "\n".join(lines)
    (args.out_dir / "diagnostic_summary.txt").write_text(summary, encoding="utf-8")

    # Plots
    s2_iou = np.asarray([r["s2_iou"] for r in rows])
    y_iou = np.asarray([r["yolo_iou"] for r in rows])

    plt.figure(figsize=(7,5))
    plt.boxplot([s2_iou, y_iou], labels=["S²A-Net","YOLO26l-OBB"], showfliers=False)
    plt.ylabel("IoU")
    plt.title("IoU sur les mêmes avions détectés à IoU ≥ 0,50")
    plt.tight_layout()
    plt.savefig(args.out_dir / "01_iou_common_gt.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.boxplot(
        [[r["s2_angle_err_deg"] for r in rows],
         [r["yolo_angle_err_deg"] for r in rows]],
        labels=["S²A-Net","YOLO26l-OBB"], showfliers=False
    )
    plt.ylabel("Erreur d'angle (degrés)")
    plt.title("Erreur d'angle sur les mêmes avions")
    plt.tight_layout()
    plt.savefig(args.out_dir / "02_angle_error_common_gt.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.boxplot(
        [[r["s2_center_err_norm"] for r in rows],
         [r["yolo_center_err_norm"] for r in rows]],
        labels=["S²A-Net","YOLO26l-OBB"], showfliers=False
    )
    plt.ylabel("Erreur de centre normalisée")
    plt.title("Erreur de centre sur les mêmes avions")
    plt.tight_layout()
    plt.savefig(args.out_dir / "03_center_error_common_gt.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.hist(diou, bins=40)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Δ IoU = YOLO - S²A-Net")
    plt.ylabel("Nombre d'avions")
    plt.title("Gain de localisation sur les mêmes GT")
    plt.tight_layout()
    plt.savefig(args.out_dir / "04_delta_iou_hist.png", dpi=180)
    plt.close()

    # Taille
    order = ["small","medium","large"]
    xs = np.arange(3)
    svals, yvals = [], []
    for b in order:
        rr = [r for r in rows if r["size_bin"] == b]
        svals.append(mean(r["s2_iou"] for r in rr))
        yvals.append(mean(r["yolo_iou"] for r in rr))
    w = 0.36
    plt.figure(figsize=(7,5))
    plt.bar(xs-w/2, svals, width=w, label="S²A-Net")
    plt.bar(xs+w/2, yvals, width=w, label="YOLO26l-OBB")
    plt.xticks(xs, order)
    plt.ylabel("IoU moyen")
    plt.xlabel("Taille GT")
    plt.title("Localisation selon la taille des avions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "05_iou_by_size.png", dpi=180)
    plt.close()

    # Orientation
    angle_groups = sorted(set(r["angle_bin"] for r in rows))
    xs = np.arange(len(angle_groups))
    sv, yv = [], []
    for b in angle_groups:
        rr = [r for r in rows if r["angle_bin"] == b]
        sv.append(mean(r["s2_iou"] for r in rr))
        yv.append(mean(r["yolo_iou"] for r in rr))
    plt.figure(figsize=(11,5))
    plt.plot(xs, sv, marker="o", label="S²A-Net")
    plt.plot(xs, yv, marker="o", label="YOLO26l-OBB")
    plt.xticks(xs, angle_groups, rotation=45, ha="right")
    plt.ylabel("IoU moyen")
    plt.xlabel("Orientation GT (degrés)")
    plt.title("Localisation selon l'orientation des avions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "06_iou_by_angle.png", dpi=180)
    plt.close()

    # Score vs IoU
    plt.figure(figsize=(7,5))
    plt.scatter([r["s2_score"] for r in rows], s2_iou, s=10, alpha=.35, label="S²A-Net")
    plt.scatter([r["yolo_score"] for r in rows], y_iou, s=10, alpha=.35, label="YOLO26l-OBB")
    plt.xlabel("Score")
    plt.ylabel("IoU")
    plt.title("Score de confiance vs qualité de localisation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "07_score_vs_iou.png", dpi=180)
    plt.close()

    print(summary)
    print()
    print(f"[OK] Résultats écrits dans: {args.out_dir}")


if __name__ == "__main__":
    main()
