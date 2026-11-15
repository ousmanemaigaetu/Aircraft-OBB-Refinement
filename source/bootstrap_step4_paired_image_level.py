#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 4
Paired original-image bootstrap for final frozen TEST AP metrics.

READ-ONLY. No inference. No threshold tuning.

Bootstrap unit:
    original TEST image (N=51)

For each replicate:
    - sample 51 original image IDs with replacement
    - use the SAME sampled image multiplicities for baseline and refined
    - repeated images are treated as independent pseudo-images
    - recompute global detection AP from the resampled image population

Metrics:
    AP50, AP75, AP80, AP85, mAP50-95
    baseline, refined, and paired difference (refined - baseline)

CI:
    percentile 95% bootstrap CI
    fixed seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)
EXPECTED_BASE = {
    "AP50": 94.2568,
    "AP75": 50.9730,
    "AP80": 23.3370,
    "AP85": 5.6492,
    "mAP50_95": 51.2632,
}
EXPECTED_REFINED = {
    "AP50": 94.2568,
    "AP75": 62.9206,
    "AP80": 38.6229,
    "AP85": 12.6512,
    "mAP50_95": 56.1504,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def poly_iou(a, b):
    # Convention EXACTE de eval_obb_val_standalone.py
    a = cv2.convexHull(np.asarray(a, np.float32).reshape(4, 2))
    b = cv2.convexHull(np.asarray(b, np.float32).reshape(4, 2))
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


def parse_predictions(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            s = line.strip().split()
            if len(s) < 10:
                continue
            rows.append({
                "row": row_idx,
                "image_id": s[0],
                "score": float(s[1]),
                "poly": np.asarray([float(x) for x in s[2:10]], dtype=np.float64).reshape(4, 2),
            })
    return rows


def parse_gt(gt_dir):
    data = {}
    for p in sorted(gt_dir.glob("*.txt")):
        useful, difficult = [], []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip().split()
            if len(s) < 9 or s[8] != "plane":
                continue
            poly = np.asarray([float(x) for x in s[:8]], dtype=np.float64).reshape(4, 2)
            difficulty = int(s[9]) if len(s) >= 10 else 0
            if difficulty == 0:
                useful.append(poly)
            else:
                difficult.append(poly)
        data[p.stem] = {"useful": useful, "difficult": difficult}
    return data


def ap101(rec, prec):
    # Convention EXACTE de eval_obb_val_standalone.py
    recall_points = np.linspace(0.0, 1.0, 101)
    vals = []
    for r in recall_points:
        mask = rec >= r
        vals.append(float(np.max(prec[mask])) if np.any(mask) else 0.0)
    return float(np.mean(vals))


def build_global_status(preds, gt, image_ids, thr):
    """
    Returns arrays in deterministic global score order:
      score, image_index, status
    status: 1 TP, 0 FP, -1 ignored(difficult)

    Matching is performed independently within each original image, with
    detections processed in descending score order.
    """
    img_to_idx = {x: i for i, x in enumerate(image_ids)}
    by_image = {x: [] for x in image_ids}
    for r in preds:
        if r["image_id"] in by_image:
            by_image[r["image_id"]].append(r)

    status_by_row = {}

    for image_id in image_ids:
        dets = sorted(by_image[image_id], key=lambda x: (-x["score"], x["row"]))
        useful = gt[image_id]["useful"]
        difficult = gt[image_id]["difficult"]
        used = set()

        for d in dets:
            best_idx = None
            best_iou = -1.0

            for j, g in enumerate(useful):
                if j in used:
                    continue
                iou = poly_iou(d["poly"], g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_idx is not None and best_iou >= thr:
                used.add(best_idx)
                status_by_row[d["row"]] = 1
                continue

            max_diff = 0.0
            for g in difficult:
                max_diff = max(max_diff, poly_iou(d["poly"], g))

            if max_diff >= thr:
                status_by_row[d["row"]] = -1
            else:
                status_by_row[d["row"]] = 0

    ordered = sorted(preds, key=lambda x: (-x["score"], x["row"]))
    image_index = np.asarray([img_to_idx[r["image_id"]] for r in ordered], dtype=np.int16)
    status = np.asarray([status_by_row[r["row"]] for r in ordered], dtype=np.int8)
    scores = np.asarray([r["score"] for r in ordered], dtype=np.float64)
    return scores, image_index, status


def ap_from_multiplicity(image_mult, image_index, status, gt_counts):
    """
    Exact image-bootstrap weighting:
    each detection from image i is repeated image_mult[i] times.
    Ignored detections contribute neither TP nor FP.
    """
    weights = image_mult[image_index].astype(np.int64)

    tpw = weights * (status == 1)
    fpw = weights * (status == 0)

    valid = (status != -1) & (weights > 0)
    if not np.any(valid):
        return 0.0

    tp = np.cumsum(tpw[valid], dtype=np.float64)
    fp = np.cumsum(fpw[valid], dtype=np.float64)

    total_gt = int(np.dot(image_mult, gt_counts))
    if total_gt <= 0:
        return 0.0

    rec = tp / float(total_gt)
    prec = tp / np.maximum(tp + fp, 1e-12)
    return 100.0 * ap101(rec, prec)


def metric_dict(ap_by_thr):
    vals = {round(float(k), 2): float(v) for k, v in ap_by_thr.items()}
    return {
        "AP50": vals[0.50],
        "AP75": vals[0.75],
        "AP80": vals[0.80],
        "AP85": vals[0.85],
        "mAP50_95": float(np.mean([vals[round(float(t), 2)] for t in IOU_THRESHOLDS])),
    }


def full_dataset_metrics(status_cache, gt_counts, n_images):
    mult = np.ones(n_images, dtype=np.int64)
    aps = {}
    for thr in IOU_THRESHOLDS:
        _, img_idx, status = status_cache[round(float(thr), 2)]
        aps[round(float(thr), 2)] = ap_from_multiplicity(mult, img_idx, status, gt_counts)
    return metric_dict(aps)


def assert_expected(actual, expected, label, tol=5e-4):
    diffs = {k: abs(actual[k] - expected[k]) for k in expected}
    mx = max(diffs.values())
    if mx > tol:
        raise RuntimeError(
            "{} full TEST metric guard failed. max abs diff={} points\n"
            "actual={}\nexpected={}\ndiffs={}".format(label, mx, actual, expected, diffs)
        )
    return diffs


def ci_summary(x):
    a = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "ci95_low": float(np.quantile(a, 0.025)),
        "ci95_high": float(np.quantile(a, 0.975)),
    }


def main():
    args = parse_args()
    if args.replicates < 5000:
        raise ValueError("replicates must be >= 5000")

    for label, p in [
        ("GT dir", args.gt_dir),
        ("baseline", args.baseline),
        ("refined", args.refined),
    ]:
        if not p.exists():
            raise FileNotFoundError("{} not found: {}".format(label, p))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = parse_gt(args.gt_dir)
    image_ids = sorted(gt.keys())
    if len(image_ids) != 51:
        raise RuntimeError("Expected 51 original TEST images, got {}".format(len(image_ids)))

    baseline = parse_predictions(args.baseline)
    refined = parse_predictions(args.refined)

    if len(baseline) != 1100 or len(refined) != 1100:
        raise RuntimeError("Expected 1100 baseline/refined prediction rows")

    for b, r in zip(baseline, refined):
        if b["image_id"] != r["image_id"] or abs(b["score"] - r["score"]) > 1e-10:
            raise RuntimeError("Baseline/refined prediction id-score correspondence failed")

    gt_counts = np.asarray([len(gt[x]["useful"]) for x in image_ids], dtype=np.int64)
    diff_counts = np.asarray([len(gt[x]["difficult"]) for x in image_ids], dtype=np.int64)

    base_cache = {}
    ref_cache = {}
    for thr in IOU_THRESHOLDS:
        key = round(float(thr), 2)
        base_cache[key] = build_global_status(baseline, gt, image_ids, float(thr))
        ref_cache[key] = build_global_status(refined, gt, image_ids, float(thr))

    # Critical evaluator guard before any bootstrap.
    full_base = full_dataset_metrics(base_cache, gt_counts, len(image_ids))
    full_ref = full_dataset_metrics(ref_cache, gt_counts, len(image_ids))

    guard_base = assert_expected(full_base, EXPECTED_BASE, "BASELINE")
    guard_ref = assert_expected(full_ref, EXPECTED_REFINED, "REFINED")

    rng = np.random.default_rng(args.seed)
    B = args.replicates
    metric_names = ["AP50", "AP75", "AP80", "AP85", "mAP50_95"]

    boot_base = {m: np.empty(B, dtype=np.float64) for m in metric_names}
    boot_ref = {m: np.empty(B, dtype=np.float64) for m in metric_names}

    # Save sampled multiplicities for exact reproducibility.
    multiplicities = np.empty((B, len(image_ids)), dtype=np.uint8)

    for b in range(B):
        sampled = rng.integers(0, len(image_ids), size=len(image_ids))
        mult = np.bincount(sampled, minlength=len(image_ids)).astype(np.int64)
        multiplicities[b] = mult.astype(np.uint8)

        aps_b = {}
        aps_r = {}

        for thr in IOU_THRESHOLDS:
            key = round(float(thr), 2)

            _, img_idx_b, status_b = base_cache[key]
            _, img_idx_r, status_r = ref_cache[key]

            aps_b[key] = ap_from_multiplicity(mult, img_idx_b, status_b, gt_counts)
            aps_r[key] = ap_from_multiplicity(mult, img_idx_r, status_r, gt_counts)

        mb = metric_dict(aps_b)
        mr = metric_dict(aps_r)

        for m in metric_names:
            boot_base[m][b] = mb[m]
            boot_ref[m][b] = mr[m]

    # Store only compact bootstrap metrics CSV, not all sampled IDs in text.
    df = pd.DataFrame({"replicate": np.arange(B, dtype=int)})
    for m in metric_names:
        df["baseline_" + m] = boot_base[m]
        df["refined_" + m] = boot_ref[m]
        df["delta_" + m] = boot_ref[m] - boot_base[m]

    csv_path = args.out_dir / "step4_paired_bootstrap_replicates.csv"
    df.to_csv(csv_path, index=False)

    np.savez_compressed(
        args.out_dir / "step4_bootstrap_image_multiplicities.npz",
        multiplicities=multiplicities,
        image_ids=np.asarray(image_ids),
        seed=np.asarray([args.seed]),
    )

    summary = {}
    for m in metric_names:
        delta = boot_ref[m] - boot_base[m]
        summary[m] = {
            "point_baseline": full_base[m],
            "point_refined": full_ref[m],
            "point_delta": full_ref[m] - full_base[m],
            "bootstrap_baseline": ci_summary(boot_base[m]),
            "bootstrap_refined": ci_summary(boot_ref[m]),
            "bootstrap_delta": ci_summary(delta),
            "fraction_delta_gt_0": float(np.mean(delta > 0.0)),
            "fraction_delta_eq_0": float(np.mean(delta == 0.0)),
            "fraction_delta_lt_0": float(np.mean(delta < 0.0)),
        }

    out = {
        "protocol": {
            "bootstrap_unit": "original TEST image",
            "n_original_images": len(image_ids),
            "n_useful_gt": int(gt_counts.sum()),
            "n_difficult_gt": int(diff_counts.sum()),
            "replicates": B,
            "seed": args.seed,
            "paired": True,
            "sampling": "51 image IDs with replacement per replicate",
            "repeated_image_handling": "each multiplicity repeats that image's GT and detections as independent pseudo-images",
            "ci": "percentile 95% bootstrap CI",
            "iou_thresholds_for_map": [round(float(x), 2) for x in IOU_THRESHOLDS],
            "no_test_tuning": True,
        },
        "files": {
            "baseline": str(args.baseline),
            "baseline_sha256": sha256(args.baseline),
            "refined": str(args.refined),
            "refined_sha256": sha256(args.refined),
        },
        "full_test_guard": {
            "baseline": full_base,
            "refined": full_ref,
            "baseline_abs_diffs_vs_expected": guard_base,
            "refined_abs_diffs_vs_expected": guard_ref,
        },
        "metrics": summary,
        "limitations": [
            "The TEST set contains only 51 original images, so image-level uncertainty remains substantial despite many object instances.",
            "Bootstrap confidence intervals quantify sampling variability with respect to the observed TEST image population; they do not prove generalization to a new domain.",
            "The paired bootstrap preserves correlation between baseline and refined results by using identical image resamples.",
        ],
    }

    json_path = args.out_dir / "step4_paired_bootstrap_full.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    txt_path = args.out_dir / "step4_paired_bootstrap_compact_summary.txt"
    lines = [
        "=" * 100,
        "STEP 4 PAIRED IMAGE-LEVEL BOOTSTRAP",
        "=" * 100,
        "unit: original TEST image",
        "n_images: {}".format(len(image_ids)),
        "useful_gt: {}".format(int(gt_counts.sum())),
        "difficult_gt: {}".format(int(diff_counts.sum())),
        "replicates: {}".format(B),
        "seed: {}".format(args.seed),
        "CI: percentile 95%",
        "",
        "FULL TEST EVALUATOR GUARD",
        "baseline: {}".format(full_base),
        "refined: {}".format(full_ref),
        "max_abs_guard_diff_baseline_points: {}".format(max(guard_base.values())),
        "max_abs_guard_diff_refined_points: {}".format(max(guard_ref.values())),
        "",
        "PAIRED BOOTSTRAP RESULTS",
    ]

    for m in metric_names:
        s = summary[m]
        d = s["bootstrap_delta"]
        lines += [
            "",
            m,
            " point baseline/refined/delta: {:.4f} / {:.4f} / {:+.4f}".format(
                s["point_baseline"], s["point_refined"], s["point_delta"]
            ),
            " baseline CI95: [{:.4f}, {:.4f}]".format(
                s["bootstrap_baseline"]["ci95_low"],
                s["bootstrap_baseline"]["ci95_high"],
            ),
            " refined CI95: [{:.4f}, {:.4f}]".format(
                s["bootstrap_refined"]["ci95_low"],
                s["bootstrap_refined"]["ci95_high"],
            ),
            " delta bootstrap median: {:+.4f}".format(d["median"]),
            " delta CI95: [{:+.4f}, {:+.4f}]".format(d["ci95_low"], d["ci95_high"]),
            " fraction delta > 0 / = 0 / < 0: {:.6f} / {:.6f} / {:.6f}".format(
                s["fraction_delta_gt_0"],
                s["fraction_delta_eq_0"],
                s["fraction_delta_lt_0"],
            ),
        ]

    lines += [
        "",
        "IMPORTANT",
        "These CIs are paired ORIGINAL-IMAGE bootstrap intervals, not object-level or patch-level intervals.",
        "N=51 original TEST images remains a limitation and must be stated in the paper.",
        "",
        "json: {}".format(json_path),
        "csv: {}".format(csv_path),
        "multiplicities_npz: {}".format(args.out_dir / "step4_bootstrap_image_multiplicities.npz"),
    ]

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(txt_path)
    print("DONE")


if __name__ == "__main__":
    main()
