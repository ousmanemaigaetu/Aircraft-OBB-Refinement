#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 1C.2
Focused audit of DOTA quadrilateral -> OBB conversion used by the project.

READ-ONLY:
- inspects project source code
- compares GT polygons to several OBB conversion conventions
- checks which convention reproduces stored gt_cx/gt_cy/gt_major/gt_minor/gt_angle_deg
- does not modify dataset, checkpoints, predictions or results
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--matches",
        type=Path,
        default=Path("runs/angle_refiner/train_matches.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/scientific_audit/step1c2_gt_obb_conversion"),
    )
    return p.parse_args()


def signed_axis_delta(a, b):
    return float(((b - a + 90.0) % 180.0) - 90.0)


def abs_axis_delta(a, b):
    return abs(signed_axis_delta(a, b))


def poly_iou(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 2)
    b = np.asarray(b, np.float32).reshape(-1, 2)
    aa = abs(float(cv2.contourArea(a)))
    ab = abs(float(cv2.contourArea(b)))
    if aa <= 0 or ab <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    inter = max(float(inter), 0.0)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def rect_poly(cx, cy, major, minor, angle_deg):
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)], dtype=np.float64)
    v = np.array([-math.sin(t), math.cos(t)], dtype=np.float64)
    c = np.array([cx, cy], dtype=np.float64)
    du = 0.5 * major * u
    dv = 0.5 * minor * v
    return np.stack([c-du-dv, c+du-dv, c+du+dv, c-du+dv], axis=0)


def conv_edge(poly):
    p = np.asarray(poly, np.float64).reshape(4, 2)
    c = p.mean(0)
    e01 = p[1] - p[0]
    e12 = p[2] - p[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))
    if l01 >= l12:
        major, minor, vec = l01, l12, e01
    else:
        major, minor, vec = l12, l01, e12
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180
    return dict(cx=float(c[0]), cy=float(c[1]), major=major, minor=minor, angle_deg=angle)


def conv_minarea(poly):
    p = np.asarray(poly, np.float32).reshape(4, 2)
    (cx, cy), (w, h), a = cv2.minAreaRect(p)

    # OpenCV angle convention depends on version. Convert by the actual
    # returned width/height axes, then canonicalize major >= minor.
    box = cv2.boxPoints(((cx, cy), (w, h), a)).astype(np.float64)

    # Recover canonical geometry from the returned rectangle itself.
    q = box
    e01 = q[1] - q[0]
    e12 = q[2] - q[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))
    if l01 >= l12:
        major, minor, vec = l01, l12, e01
    else:
        major, minor, vec = l12, l01, e12
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180
    return dict(cx=float(cx), cy=float(cy), major=major, minor=minor, angle_deg=angle)


def conv_pca(poly):
    p = np.asarray(poly, np.float64).reshape(4, 2)
    c = p.mean(0)
    x = p - c
    cov = x.T @ x / len(x)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, np.argmax(vals)]
    angle = math.degrees(math.atan2(float(u[1]), float(u[0]))) % 180

    # project polygon vertices onto PCA axes and take bounding extents
    v = np.array([-u[1], u[0]])
    pu = x @ u
    pv = x @ v
    major = float(pu.max() - pu.min())
    minor = float(pv.max() - pv.min())
    if minor > major:
        major, minor = minor, major
        angle = (angle + 90.0) % 180
    return dict(cx=float(c[0]), cy=float(c[1]), major=major, minor=minor, angle_deg=angle)


def errors(pred, row):
    return {
        "center": math.hypot(pred["cx"] - row.gt_cx, pred["cy"] - row.gt_cy),
        "major": abs(pred["major"] - row.gt_major),
        "minor": abs(pred["minor"] - row.gt_minor),
        "angle": abs_axis_delta(pred["angle_deg"], row.gt_angle_deg),
    }


def summary(x):
    a = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q90": float(np.quantile(a, 0.90)),
        "q95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def source_hits(path, patterns, radius=4):
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    regs = [re.compile(p, re.I) for p in patterns]
    idx = [i for i, line in enumerate(lines) if any(rx.search(line) for rx in regs)]
    keep = set()
    for i in idx:
        for j in range(max(0, i-radius), min(len(lines), i+radius+1)):
            keep.add(j)
    return [{"line": i+1, "text": lines[i]} for i in sorted(keep)]


def main():
    args = parse_args()
    root = args.project_root.resolve()
    matches = (root / args.matches).resolve() if not args.matches.is_absolute() else args.matches.resolve()
    out = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(matches)

    methods = {
        "edge_based": conv_edge,
        "opencv_minAreaRect": conv_minarea,
        "pca_bbox": conv_pca,
    }

    all_rows = []
    agg = {m: {k: [] for k in ("center","major","minor","angle","poly_iou")} for m in methods}

    for row in df.itertuples(index=False):
        gt = np.array(
            [[getattr(row, f"gt_x{i}"), getattr(row, f"gt_y{i}")] for i in range(1,5)],
            dtype=np.float64,
        )
        rec = {"image_id": row.image_id, "det_rank": row.det_rank}
        for name, fn in methods.items():
            g = fn(gt)
            e = errors(g, row)
            rp = rect_poly(g["cx"], g["cy"], g["major"], g["minor"], g["angle_deg"])
            iou = poly_iou(gt, rp)
            for k,v in e.items():
                rec[f"{name}_{k}_err"] = v
                agg[name][k].append(v)
            rec[f"{name}_poly_iou"] = iou
            agg[name]["poly_iou"].append(iou)
        all_rows.append(rec)

    detail = pd.DataFrame(all_rows)
    detail.to_csv(out / "gt_conversion_comparison.csv", index=False)

    comparisons = {}
    for name in methods:
        comparisons[name] = {
            "center_error_px": summary(agg[name]["center"]),
            "major_error_px": summary(agg[name]["major"]),
            "minor_error_px": summary(agg[name]["minor"]),
            "angle_error_deg": summary(agg[name]["angle"]),
            "raw_polygon_vs_reconstructed_rect_iou": summary(agg[name]["poly_iou"]),
            "exactish_center_lt_1e3_px_fraction": float(np.mean(np.array(agg[name]["center"]) < 1e-3)),
            "exactish_angle_lt_1e2_deg_fraction": float(np.mean(np.array(agg[name]["angle"]) < 1e-2)),
        }

    # Inspect actual project sources around GT conversion.
    source_files = [
        root / "build_angle_refiner_dataset.py",
        root / "run_obb_angle_refiner.py",
        root / "eval_obb_val_standalone.py",
        root / "modeles_reference/s2anet/tools/prepare_avions_s2anet.py",
        root / "modeles_modernes/yolo26_obb/run_yolo26_obb_full.py",
    ]
    patterns = [
        r"minAreaRect", r"boxPoints", r"poly2obb", r"obb2poly",
        r"gt_angle", r"major", r"minor", r"angle_deg",
        r"polygon", r"intersectConvexConvex"
    ]
    evidence = {}
    for p in source_files:
        evidence[str(p)] = source_hits(p, patterns, radius=5)

    with (out / "source_conversion_evidence.txt").open("w", encoding="utf-8") as f:
        for p, rows in evidence.items():
            f.write("="*100 + "\n")
            f.write(p + "\n")
            f.write("="*100 + "\n")
            for r in rows:
                f.write(f"L{r['line']}: {r['text']}\n")
            f.write("\n")

    # Identify which candidate best reproduces the stored labels.
    ranking = []
    for name, s in comparisons.items():
        # scale-free-ish score: median angle + center + normalized dimensions
        med_center = s["center_error_px"]["median"]
        med_angle = s["angle_error_deg"]["median"]
        med_major = s["major_error_px"]["median"]
        med_minor = s["minor_error_px"]["median"]
        score = med_angle + med_center + med_major + med_minor
        ranking.append((score, name))
    ranking.sort()

    full = {
        "rows": len(df),
        "candidate_conversion_comparisons": comparisons,
        "candidate_ranking_by_simple_median_error_sum": [name for _,name in ranking],
        "best_candidate_by_this_diagnostic": ranking[0][1],
        "important_note": (
            "Raw DOTA quadrilaterals are not assumed to be exact rectangles. "
            "The relevant question is which conversion reproduces the stored refiner labels "
            "and what the project source code actually implements."
        ),
        "source_evidence_file": str(out / "source_conversion_evidence.txt"),
    }
    (out / "audit_step1c2_full.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    lines = []
    lines += ["="*92, "STEP 1C.2 GT QUADRILATERAL -> OBB CONVERSION", "="*92]
    lines.append(f"rows: {len(df)}")
    for name in methods:
        s = comparisons[name]
        lines += [
            "",
            f"METHOD: {name}",
            f" center median/max px: {s['center_error_px']['median']:.6f} / {s['center_error_px']['max']:.6f}",
            f" major median/max px : {s['major_error_px']['median']:.6f} / {s['major_error_px']['max']:.6f}",
            f" minor median/max px : {s['minor_error_px']['median']:.6f} / {s['minor_error_px']['max']:.6f}",
            f" angle median/max deg: {s['angle_error_deg']['median']:.6f} / {s['angle_error_deg']['max']:.6f}",
            f" polygon-rect IoU median/min: {s['raw_polygon_vs_reconstructed_rect_iou']['median']:.6f} / {s['raw_polygon_vs_reconstructed_rect_iou']['min']:.6f}",
            f" center <1e-3 fraction: {s['exactish_center_lt_1e3_px_fraction']:.6f}",
            f" angle <1e-2 fraction : {s['exactish_angle_lt_1e2_deg_fraction']:.6f}",
        ]

    lines += [
        "",
        "RANKING:",
        " -> ".join(name for _,name in ranking),
        "",
        "SOURCE EVIDENCE FILE:",
        str(out / "source_conversion_evidence.txt"),
        "",
        "IMPORTANT:",
        "A low raw-polygon vs rectangle IoU is not itself a bug if DOTA GT is an arbitrary quadrilateral.",
        "The stored-label reproduction and source implementation determine the correct convention."
    ]

    compact = out / "audit_step1c2_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")

    print(compact)
    print(out / "source_conversion_evidence.txt")


if __name__ == "__main__":
    main()
