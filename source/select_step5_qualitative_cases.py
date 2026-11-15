#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 5
Objective qualitative case selection and visualization for final frozen TEST.

READ-ONLY. No inference. No tuning.

Selection is deterministic from Step 3 per-match geometry:
  A) largest IoU improvements
  B) largest IoU degradations
  C) IoU 0.75 upward crossings
  D) IoU 0.75 downward crossings
  E) small-object (<32 px) degradations
  F) near-square ([1.0,1.1)) cases with large gains

This avoids manual cherry-picking. The script exports:
  - candidate_cases.csv
  - selected_cases_summary.txt
  - individual PNG crops
  - multi-panel PNG and PDF contact sheets

Overlay:
  GT       = green
  S2A-Net  = red
  Refined  = blue
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--step3-csv", type=Path,
        default=Path("runs/final_test_geometry/step3/step3_test_geometry_per_match.csv")
    )
    p.add_argument(
        "--images", type=Path,
        default=Path("dataset_avions/test/images")
    )
    p.add_argument(
        "--gt-dir", type=Path,
        default=Path("dataset_avions/test/annotations")
    )
    p.add_argument(
        "--baseline", type=Path,
        default=Path("modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/test_final/results_after_nms/plane.txt")
    )
    p.add_argument(
        "--refined", type=Path,
        default=Path("runs/angle_refiner/resnet18_angle/test_final_frozen_clamp60/refined_plane_frozen_clamp60.txt")
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("runs/qualitative/step5")
    )
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def parse_predictions(path):
    rows = {}
    line_idx = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip().split()
        if len(s) < 10:
            line_idx += 1
            continue
        rows[line_idx] = {
            "image_id": s[0],
            "score": float(s[1]),
            "poly": np.asarray(list(map(float, s[2:10])), dtype=np.float32).reshape(4, 2)
        }
        line_idx += 1
    return rows


def parse_gt(gt_dir):
    out = {}
    for p in sorted(gt_dir.glob("*.txt")):
        useful = []
        difficult = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip().split()
            if len(s) < 9 or s[8] != "plane":
                continue
            poly = np.asarray(list(map(float, s[:8])), dtype=np.float32).reshape(4, 2)
            difficulty = int(float(s[9])) if len(s) >= 10 else 0
            if difficulty == 0:
                useful.append(poly)
            else:
                difficult.append(poly)
        out[p.stem] = {"useful": useful, "difficult": difficult}
    return out


def image_map(folder):
    m = {}
    for p in folder.iterdir():
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            m[p.stem] = p
    return m


def union_crop_bounds(polys, image_shape, pad_frac=0.35, min_side=220):
    pts = np.concatenate([np.asarray(x).reshape(-1, 2) for x in polys], axis=0)
    x0, y0 = np.min(pts, axis=0)
    x1, y1 = np.max(pts, axis=0)

    w = max(float(x1 - x0), 1.0)
    h = max(float(y1 - y0), 1.0)
    side = max(w, h)
    side = max(side * (1.0 + 2.0 * pad_frac), float(min_side))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    H, W = image_shape[:2]
    xa = int(math.floor(cx - side / 2))
    ya = int(math.floor(cy - side / 2))
    xb = int(math.ceil(cx + side / 2))
    yb = int(math.ceil(cy + side / 2))

    xa = max(0, xa)
    ya = max(0, ya)
    xb = min(W, xb)
    yb = min(H, yb)

    return xa, ya, xb, yb


def draw_poly(img, poly, color, thickness=3):
    q = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [q], True, color, thickness, lineType=cv2.LINE_AA)


def build_case_image(img_bgr, gt_poly, base_poly, ref_poly, title_lines):
    canvas = img_bgr.copy()

    # GT = green
    draw_poly(canvas, gt_poly, (0, 200, 0), 4)

    # S2A-Net = red
    draw_poly(canvas, base_poly, (0, 0, 230), 3)

    # Refined = blue
    draw_poly(canvas, ref_poly, (230, 80, 0), 3)

    xa, ya, xb, yb = union_crop_bounds(
        [gt_poly, base_poly, ref_poly],
        canvas.shape
    )

    crop = canvas[ya:yb, xa:xb].copy()

    # Aucun texte dans la vignette.
    # Les métriques seront placées dans la légende de la figure finale.
    return crop


def select_cases(df, k):
    selections = []

    def take(name, sub, sort_col, ascending):
        if len(sub) == 0:
            return
        x = sub.sort_values(sort_col, ascending=ascending).head(k).copy()
        for rank, (_, row) in enumerate(x.iterrows(), start=1):
            rec = row.to_dict()
            rec["selection_group"] = name
            rec["selection_rank"] = rank
            selections.append(rec)

    take(
        "largest_iou_improvement",
        df,
        "iou_gain",
        False
    )

    take(
        "largest_iou_degradation",
        df,
        "iou_gain",
        True
    )

    take(
        "cross_up_075",
        df[(df["iou_baseline"] < 0.75) & (df["iou_refined"] >= 0.75)],
        "iou_gain",
        False
    )

    take(
        "cross_down_075",
        df[(df["iou_baseline"] >= 0.75) & (df["iou_refined"] < 0.75)],
        "iou_gain",
        True
    )

    take(
        "small_object_degradation",
        df[(df["size_bin"] == "<32") & (df["iou_gain"] < 0)],
        "iou_gain",
        True
    )

    take(
        "near_square_large_gain",
        df[df["aspect_bin"] == "[1.0,1.1)"],
        "iou_gain",
        False
    )

    out = pd.DataFrame(selections)
    if len(out):
        out = out.drop_duplicates(
            subset=["selection_group", "image_id", "prediction_row", "gt_index"]
        ).reset_index(drop=True)
    return out


def main():
    args = parse_args()
    for label, p in [
        ("step3 csv", args.step3_csv),
        ("images", args.images),
        ("gt dir", args.gt_dir),
        ("baseline", args.baseline),
        ("refined", args.refined),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    indiv_dir = args.out_dir / "individual"
    indiv_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.step3_csv)
    required = {
        "image_id", "prediction_row", "gt_index",
        "iou_baseline", "iou_refined", "iou_gain",
        "angle_error_baseline_deg", "angle_error_refined_deg",
        "size_bin", "aspect_bin"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing Step3 columns: {missing}")

    selected = select_cases(df, args.top_k)
    if len(selected) == 0:
        raise RuntimeError("No qualitative cases selected")

    selected.to_csv(args.out_dir / "candidate_cases.csv", index=False)

    base = parse_predictions(args.baseline)
    ref = parse_predictions(args.refined)
    gt = parse_gt(args.gt_dir)
    imgs = image_map(args.images)

    manifest = []
    group_images = {}

    for idx, row in selected.iterrows():
        image_id = str(row["image_id"])
        pred_row = int(row["prediction_row"])
        gt_idx = int(row["gt_index"])

        if image_id not in imgs:
            raise RuntimeError(f"Image not found for {image_id}")
        if pred_row not in base or pred_row not in ref:
            raise RuntimeError(f"Prediction row {pred_row} not found")
        if image_id not in gt or gt_idx >= len(gt[image_id]["useful"]):
            raise RuntimeError(f"GT index invalid: {image_id}, {gt_idx}")

        img = cv2.imread(str(imgs[image_id]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read {imgs[image_id]}")

        gt_poly = gt[image_id]["useful"][gt_idx]
        base_poly = base[pred_row]["poly"]
        ref_poly = ref[pred_row]["poly"]

        title_lines = [
            f"IoU {row['iou_baseline']:.3f} -> {row['iou_refined']:.3f} ({row['iou_gain']:+.3f})",
            f"angle err {row['angle_error_baseline_deg']:.1f} -> {row['angle_error_refined_deg']:.1f} deg",
            f"size {row['size_bin']} | aspect {row['aspect_bin']}",
        ]

        crop = build_case_image(
            img, gt_poly, base_poly, ref_poly, title_lines
        )

        group = str(row["selection_group"])
        rank = int(row["selection_rank"])
        fname = f"{group}_{rank:02d}_{image_id}_pred{pred_row}.png"
        out_path = indiv_dir / fname
        cv2.imwrite(str(out_path), crop)

        manifest.append({
            "selection_group": group,
            "selection_rank": rank,
            "image_id": image_id,
            "prediction_row": pred_row,
            "gt_index": gt_idx,
            "iou_baseline": float(row["iou_baseline"]),
            "iou_refined": float(row["iou_refined"]),
            "iou_gain": float(row["iou_gain"]),
            "angle_error_baseline_deg": float(row["angle_error_baseline_deg"]),
            "angle_error_refined_deg": float(row["angle_error_refined_deg"]),
            "size_bin": str(row["size_bin"]),
            "aspect_bin": str(row["aspect_bin"]),
            "png": str(out_path),
        })

        group_images.setdefault(group, []).append((rank, crop, row))

    pd.DataFrame(manifest).to_csv(args.out_dir / "selected_cases_manifest.csv", index=False)

    # One deterministic contact sheet per selection group.
    for group, items in group_images.items():
        items = sorted(items, key=lambda x: x[0])
        n = len(items)

        fig, axes = plt.subplots(1, n, figsize=(5.2*n, 5.2), squeeze=False)
        for j, (rank, crop, row) in enumerate(items):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[0, j].imshow(rgb)
            axes[0, j].set_title(
                f"{rank}. {row['image_id']}\n"
                f"IoU {row['iou_baseline']:.3f} -> {row['iou_refined']:.3f}",
                fontsize=10
            )
            axes[0, j].axis("off")

        fig.suptitle(group.replace("_", " "), fontsize=13)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{group}.png", dpi=250, bbox_inches="tight")
        fig.savefig(args.out_dir / f"{group}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Compact audit summary.
    lines = [
        "=" * 100,
        "STEP 5 OBJECTIVE QUALITATIVE SELECTION",
        "=" * 100,
        f"source_step3_csv: {args.step3_csv}",
        f"top_k_per_group: {args.top_k}",
        f"selected_rows_total: {len(selected)}",
        "",
        "Selection rules are deterministic and based only on frozen TEST geometry.",
        "No case was manually chosen before ranking.",
        "",
    ]

    for group in sorted(group_images):
        sub = pd.DataFrame([x for x in manifest if x["selection_group"] == group])
        lines.append(group)
        for _, r in sub.sort_values("selection_rank").iterrows():
            lines.append(
                " rank={} image={} pred={} IoU {:.4f}->{:.4f} gain={:+.4f} "
                "angle {:.3f}->{:.3f} size={} aspect={}".format(
                    int(r["selection_rank"]), r["image_id"], int(r["prediction_row"]),
                    r["iou_baseline"], r["iou_refined"], r["iou_gain"],
                    r["angle_error_baseline_deg"], r["angle_error_refined_deg"],
                    r["size_bin"], r["aspect_bin"]
                )
            )
        lines.append("")

    summary = args.out_dir / "selected_cases_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print(summary)
    print("DONE")


if __name__ == "__main__":
    main()
