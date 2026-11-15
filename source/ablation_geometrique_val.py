#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation géométrique contrefactuelle sur VAL uniquement.

Sur les GT détectés par S²A-Net et YOLO à IoU >= 0.50:
- on représente chaque OBB par centre, longueur, largeur, angle;
- pour S²A-Net, on remplace UNE composante à la fois par la composante GT:
    centre corrigé
    angle corrigé
    dimensions corrigées
    centre+angle
    centre+dimensions
    angle+dimensions
    toutes corrigées
- même analyse pour YOLO.

Le but est de mesurer quelle composante géométrique explique le plus
la perte de localisation stricte. Aucun entraînement, aucun TEST.

Dépendances: numpy, opencv-python, matplotlib
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

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
    return 0.0 if union <= 0 else max(0.0, min(1.0, inter / union))


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
    return {
        "cx": float(cx),
        "cy": float(cy),
        "major": float(major),
        "minor": float(minor),
        "angle": float(angle),
    }


def rect_to_poly(f):
    cx, cy = f["cx"], f["cy"]
    L, W = f["major"], f["minor"]
    th = math.radians(f["angle"])

    u = np.array([math.cos(th), math.sin(th)], dtype=np.float32)
    v = np.array([-math.sin(th), math.cos(th)], dtype=np.float32)

    c = np.array([cx, cy], dtype=np.float32)
    du = 0.5 * L * u
    dv = 0.5 * W * v

    return np.stack([
        c - du - dv,
        c + du - dv,
        c + du + dv,
        c - du + dv,
    ]).astype(np.float32)


def read_gt(gt_dir, class_name="plane"):
    gt = {}
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
                poly = np.asarray(list(map(float, p[:8])), np.float32).reshape(4,2)
            except Exception:
                continue
            difficult = 0
            if len(p) >= 10:
                try:
                    difficult = int(float(p[9]))
                except Exception:
                    difficult = 0
            arr.append({"poly": poly, "difficult": difficult})
        gt[norm_id(fp.name)] = arr
    return gt


def read_preds(path):
    out = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.strip().split()
        if len(p) < 10:
            continue
        try:
            image_id = norm_id(p[0])
            score = float(p[1])
            poly = np.asarray(list(map(float, p[2:10])), np.float32).reshape(4,2)
        except Exception:
            continue
        out[image_id].append({"score": score, "poly": poly})
    for k in out:
        out[k].sort(key=lambda z: z["score"], reverse=True)
    return dict(out)


def standard_match(gt_items, pred_items, thr=0.50):
    useful = [i for i,g in enumerate(gt_items) if not g["difficult"]]
    used = set()
    matches = {}

    for pred in pred_items:
        best_i, best_iou = None, -1.0
        for i in useful:
            if i in used:
                continue
            iou = poly_iou(pred["poly"], gt_items[i]["poly"])
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_i is not None and best_iou >= thr:
            used.add(best_i)
            matches[best_i] = (pred, float(best_iou))
    return matches


def hybrid(pred_f, gt_f, use_center=False, use_angle=False, use_size=False):
    x = dict(pred_f)
    if use_center:
        x["cx"], x["cy"] = gt_f["cx"], gt_f["cy"]
    if use_angle:
        x["angle"] = gt_f["angle"]
    if use_size:
        x["major"], x["minor"] = gt_f["major"], gt_f["minor"]
    return x


CONFIGS = [
    ("baseline", False, False, False),
    ("centre", True, False, False),
    ("angle", False, True, False),
    ("dimensions", False, False, True),
    ("centre+angle", True, True, False),
    ("centre+dimensions", True, False, True),
    ("angle+dimensions", False, True, True),
    ("tout_corrige", True, True, True),
]


def summarize(vals):
    a = np.asarray(vals, float)
    return {
        "n": len(a),
        "iou_mean": float(np.mean(a)),
        "iou_median": float(np.median(a)),
        "iou_p25": float(np.percentile(a,25)),
        "iou_p75": float(np.percentile(a,75)),
        "pct_ge_075": 100.0 * float(np.mean(a >= 0.75)),
        "pct_ge_080": 100.0 * float(np.mean(a >= 0.80)),
        "pct_ge_085": 100.0 * float(np.mean(a >= 0.85)),
        "pct_ge_090": 100.0 * float(np.mean(a >= 0.90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--s2-pred", type=Path, required=True)
    ap.add_argument("--yolo-pred", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--class-name", default="plane")
    args = ap.parse_args()

    for p in [args.gt_dir, args.s2_pred, args.yolo_pred, args.out_dir]:
        parts = [x.lower() for x in Path(p).parts]
        if "test" in parts or any(x.startswith("test_") for x in parts):
            raise SystemExit(f"[STOP] Chemin TEST détecté: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = read_gt(args.gt_dir, args.class_name)
    s2p = read_preds(args.s2_pred)
    yop = read_preds(args.yolo_pred)

    results = {
        "S2A-Net": {name: [] for name,_,_,_ in CONFIGS},
        "YOLO26l-OBB": {name: [] for name,_,_,_ in CONFIGS},
    }

    pair_rows = []

    for image_id, gt_items in gt.items():
        sm = standard_match(gt_items, s2p.get(image_id, []), 0.50)
        ym = standard_match(gt_items, yop.get(image_id, []), 0.50)

        common = sorted(set(sm) & set(ym))

        for gi in common:
            gt_poly = gt_items[gi]["poly"]
            gt_f = rect_features(gt_poly)
            gt_rect = rect_to_poly(gt_f)

            for model, match in [("S2A-Net", sm[gi]), ("YOLO26l-OBB", ym[gi])]:
                pred, original_iou = match
                pred_f = rect_features(pred["poly"])

                row = {
                    "image_id": image_id,
                    "gt_idx": gi,
                    "model": model,
                    "original_polygon_iou": original_iou,
                }

                for name, c, a, s in CONFIGS:
                    h = hybrid(pred_f, gt_f, c, a, s)
                    hp = rect_to_poly(h)

                    # Contrefactuel calculé dans la représentation rectangle canonique.
                    iou = poly_iou(hp, gt_rect)
                    results[model][name].append(iou)
                    row[name] = iou

                pair_rows.append(row)

    # CSV détaillé
    fields = list(pair_rows[0].keys())
    with open(args.out_dir/"counterfactual_per_object.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair_rows)

    # Résumé
    summary_rows = []
    for model in ["S2A-Net","YOLO26l-OBB"]:
        base_med = np.median(results[model]["baseline"])
        base_pass = 100*np.mean(np.asarray(results[model]["baseline"]) >= .75)

        for name,_,_,_ in CONFIGS:
            s = summarize(results[model][name])
            s["model"] = model
            s["config"] = name
            s["delta_median_vs_baseline"] = s["iou_median"] - base_med
            s["delta_pass75_pts"] = s["pct_ge_075"] - base_pass
            summary_rows.append(s)

    with open(args.out_dir/"counterfactual_summary.csv","w",newline="",encoding="utf-8") as f:
        fields2 = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(summary_rows)

    # Texte
    lines = []
    lines.append("ABLATION GEOMETRIQUE CONTREFACTUELLE - VAL")
    lines.append("="*78)
    lines.append("Analyse sur les mêmes GT détectés par les deux modèles à IoU >= 0.50.")
    lines.append("Les IoU contrefactuels utilisent la représentation rectangle canonique.")
    lines.append("Aucun réglage sur TEST.")
    lines.append("")

    for model in ["S2A-Net","YOLO26l-OBB"]:
        lines.append(model)
        lines.append("-"*78)
        lines.append(
            f"{'Configuration':22s} {'IoU médian':>11s} {'Δ médian':>10s} "
            f"{'>=0.75':>9s} {'Δ pass75':>10s}"
        )
        for r in [x for x in summary_rows if x["model"]==model]:
            lines.append(
                f"{r['config']:22s} {r['iou_median']:11.4f} "
                f"{r['delta_median_vs_baseline']:+10.4f} "
                f"{r['pct_ge_075']:8.2f}% "
                f"{r['delta_pass75_pts']:+9.2f} pts"
            )
        lines.append("")

    # Classement unitaire pour S2
    s2single = [
        r for r in summary_rows
        if r["model"]=="S2A-Net" and r["config"] in {"centre","angle","dimensions"}
    ]
    s2single = sorted(s2single, key=lambda r:r["delta_pass75_pts"], reverse=True)

    lines.append("CLASSEMENT DES CORRECTIONS UNITAIRES S2A-NET")
    lines.append("-"*78)
    for i,r in enumerate(s2single,1):
        lines.append(
            f"{i}. {r['config']}: "
            f"{r['delta_pass75_pts']:+.2f} points d'objets franchissant IoU 0.75, "
            f"ΔIoU médian {r['delta_median_vs_baseline']:+.4f}"
        )

    text = "\n".join(lines)
    (args.out_dir/"counterfactual_summary.txt").write_text(text, encoding="utf-8")
    print(text)

    # Figure S2 - syntaxe compatible Matplotlib récent
    s2rows = [r for r in summary_rows if r["model"]=="S2A-Net"]
    names = [r["config"] for r in s2rows]
    passes = [r["pct_ge_075"] for r in s2rows]

    plt.figure(figsize=(11,5))
    plt.bar(np.arange(len(names)), passes)
    plt.xticks(np.arange(len(names)), names, rotation=35, ha="right")
    plt.ylabel("Objets avec IoU >= 0,75 (%)")
    plt.title("S²A-Net - effet des corrections géométriques contrefactuelles")
    plt.tight_layout()
    plt.savefig(args.out_dir/"01_s2_counterfactual_pass75.png", dpi=180)
    plt.close()

    # Figure gains unitaires
    single_names = [r["config"] for r in s2single]
    single_gain = [r["delta_pass75_pts"] for r in s2single]

    plt.figure(figsize=(7,5))
    plt.bar(np.arange(len(single_names)), single_gain)
    plt.xticks(np.arange(len(single_names)), single_names)
    plt.ylabel("Gain du taux IoU >= 0,75 (points)")
    plt.title("S²A-Net - contribution géométrique unitaire")
    plt.tight_layout()
    plt.savefig(args.out_dir/"02_s2_single_component_gain.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
