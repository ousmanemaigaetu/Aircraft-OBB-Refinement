#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 6
Publication-ready quantitative figures and source tables.

READ-ONLY.
No model inference.
No TEST tuning.
No metric recomputation from predictions.

Inputs are only previously validated outputs:
- Step 2B compact summaries
- Step 3 per-match CSV
- Step 4 paired bootstrap CSV

Exports:
- vector PDF + SVG figures
- CSV source tables
- compact manifest
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Project palette already selected for the paper.
C1 = "#0649B8"
C2 = "#337EDA"
C3 = "#8DBAF2"

YOLO_TEST = {
    "AP50": 96.49297,
    "AP75": 84.44985,
    "mAP50_95": 68.8126,
}

S2_TEST = {
    "AP50": 94.2568,
    "AP75": 50.9730,
    "mAP50_95": 51.2632,
}

REF_TEST = {
    "AP50": 94.2568,
    "AP75": 62.9206,
    "mAP50_95": 56.1504,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--step3-csv",
        type=Path,
        default=Path("runs/final_test_geometry/step3/step3_test_geometry_per_match.csv"),
    )
    p.add_argument(
        "--step4-csv",
        type=Path,
        default=Path("runs/bootstrap/step4_paired_image_level/step4_paired_bootstrap_replicates.csv"),
    )
    p.add_argument(
        "--step2b1-summary",
        type=Path,
        default=Path("runs/computational_characterization/step2b1_yolo_refiner/step2b1_compact_summary.txt"),
    )
    p.add_argument(
        "--step2b2-summary",
        type=Path,
        default=Path("runs/computational_characterization/step2b2_s2anet/step2b2_compact_summary.txt"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/publication/step6"),
    )
    return p.parse_args()


def save_vector(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def q95(x):
    return np.quantile(np.asarray(x, dtype=float), [0.025, 0.975])


def safe_mean(x):
    return float(np.mean(np.asarray(x, dtype=float)))


def extract_number(text, label):
    # Simple parser for already-generated compact summaries.
    pat = re.escape(label) + r":\s*([0-9eE+\-.]+)"
    m = re.search(pat, text)
    return float(m.group(1)) if m else None


def main():
    args = parse_args()

    for label, p in [
        ("Step 3 CSV", args.step3_csv),
        ("Step 4 CSV", args.step4_csv),
        ("Step 2B.1 summary", args.step2b1_summary),
        ("Step 2B.2 summary", args.step2b2_summary),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = args.out_dir / "figures"
    tables = args.out_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    step3 = pd.read_csv(args.step3_csv)
    step4 = pd.read_csv(args.step4_csv)

    # ------------------------------------------------------------
    # TABLE 1 - Main TEST performance
    # ------------------------------------------------------------
    main_metrics = pd.DataFrame([
        {
            "method": "S2A-Net",
            "AP50": S2_TEST["AP50"],
            "AP75": S2_TEST["AP75"],
            "mAP50_95": S2_TEST["mAP50_95"],
        },
        {
            "method": "S2A-Net + angle refiner",
            "AP50": REF_TEST["AP50"],
            "AP75": REF_TEST["AP75"],
            "mAP50_95": REF_TEST["mAP50_95"],
        },
        {
            "method": "YOLO26l-OBB",
            "AP50": YOLO_TEST["AP50"],
            "AP75": YOLO_TEST["AP75"],
            "mAP50_95": YOLO_TEST["mAP50_95"],
        },
    ])
    main_metrics.to_csv(tables / "table_main_test_metrics.csv", index=False)

    # Figure 1: Main TEST performance.
    labels = ["AP50", "AP75", "mAP50-95"]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(
        x - width,
        [S2_TEST["AP50"], S2_TEST["AP75"], S2_TEST["mAP50_95"]],
        width,
        label="S2A-Net",
        color=C1,
    )
    ax.bar(
        x,
        [REF_TEST["AP50"], REF_TEST["AP75"], REF_TEST["mAP50_95"]],
        width,
        label="S2A-Net + refiner",
        color=C2,
    )
    ax.bar(
        x + width,
        [YOLO_TEST["AP50"], YOLO_TEST["AP75"], YOLO_TEST["mAP50_95"]],
        width,
        label="YOLO26l-OBB",
        color=C3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AP (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    save_vector(fig, figures / "fig_main_test_performance")

    # ------------------------------------------------------------
    # TABLE 2 - Paired bootstrap confidence intervals
    # ------------------------------------------------------------
    bootstrap_metrics = [
        ("AP75", "delta_AP75"),
        ("AP80", "delta_AP80"),
        ("AP85", "delta_AP85"),
        ("mAP50-95", "delta_mAP50_95"),
    ]

    boot_rows = []
    for label, col in bootstrap_metrics:
        if col not in step4.columns:
            raise RuntimeError(f"Missing bootstrap column: {col}")
        vals = step4[col].to_numpy(float)
        lo, hi = q95(vals)
        boot_rows.append({
            "metric": label,
            "point_delta": {
                "AP75": REF_TEST["AP75"] - S2_TEST["AP75"],
                "AP80": 15.2859,
                "AP85": 7.0020,
                "mAP50-95": REF_TEST["mAP50_95"] - S2_TEST["mAP50_95"],
            }[label],
            "bootstrap_median": float(np.median(vals)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "fraction_positive": float(np.mean(vals > 0)),
        })

    boot_table = pd.DataFrame(boot_rows)
    boot_table.to_csv(tables / "table_paired_bootstrap_ci.csv", index=False)

    # Figure 2: point gains and paired 95% CI.
    y = np.arange(len(boot_table))
    point = boot_table["point_delta"].to_numpy(float)
    lo = boot_table["ci95_low"].to_numpy(float)
    hi = boot_table["ci95_high"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    ax.errorbar(
        point,
        y,
        xerr=np.vstack([point - lo, hi - point]),
        fmt="o",
        capsize=5,
        linewidth=1.8,
        markersize=7,
        color=C1,
    )
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(boot_table["metric"])
    ax.set_xlabel("Paired improvement (percentage points)")
    ax.set_title("Original-image paired bootstrap, 95% CI")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    save_vector(fig, figures / "fig_paired_bootstrap_ci")

    # ------------------------------------------------------------
    # TABLE 3 - Geometry summary
    # ------------------------------------------------------------
    geom = pd.DataFrame([
        {
            "quantity": "Angular error (deg)",
            "baseline_mean": safe_mean(step3["angle_error_baseline_deg"]),
            "baseline_median": float(np.median(step3["angle_error_baseline_deg"])),
            "refined_mean": safe_mean(step3["angle_error_refined_deg"]),
            "refined_median": float(np.median(step3["angle_error_refined_deg"])),
        },
        {
            "quantity": "IoU",
            "baseline_mean": safe_mean(step3["iou_baseline"]),
            "baseline_median": float(np.median(step3["iou_baseline"])),
            "refined_mean": safe_mean(step3["iou_refined"]),
            "refined_median": float(np.median(step3["iou_refined"])),
        },
    ])
    geom.to_csv(tables / "table_geometry_summary.csv", index=False)

    # Figure 3: paired geometry distributions as compact boxplots.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    data = [
        step3["angle_error_baseline_deg"].to_numpy(float),
        step3["angle_error_refined_deg"].to_numpy(float),
    ]
    bp = ax.boxplot(
        data,
        tick_labels=["S2A-Net", "Refined"],
        showfliers=False,
        widths=0.55,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor(C1)
    bp["boxes"][1].set_facecolor(C2)
    ax.set_ylabel("Absolute angular error (deg)")
    ax.set_title("Matched TEST true positives, n = {}".format(len(step3)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_vector(fig, figures / "fig_angle_error_distribution")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    data = [
        step3["iou_baseline"].to_numpy(float),
        step3["iou_refined"].to_numpy(float),
    ]
    bp = ax.boxplot(
        data,
        tick_labels=["S2A-Net", "Refined"],
        showfliers=False,
        widths=0.55,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor(C1)
    bp["boxes"][1].set_facecolor(C2)
    ax.set_ylabel("IoU")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Matched TEST true positives, n = {}".format(len(step3)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_vector(fig, figures / "fig_iou_distribution")

    # ------------------------------------------------------------
    # TABLE 4 - Size-bin conditional pass75
    # ------------------------------------------------------------
    size_order = ["<32", "[32,64)", "[64,128)", ">=128"]
    rows = []
    for b in size_order:
        sub = step3[step3["size_bin"] == b]
        if len(sub) == 0:
            continue
        base = 100.0 * float(np.mean(sub["iou_baseline"] >= 0.75))
        ref = 100.0 * float(np.mean(sub["iou_refined"] >= 0.75))
        rows.append({
            "size_bin": b,
            "n": int(len(sub)),
            "baseline_pass75_pct": base,
            "refined_pass75_pct": ref,
            "delta_pass75_points": ref - base,
        })

    size_table = pd.DataFrame(rows)
    size_table.to_csv(tables / "table_size_bin_pass75.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(size_table))
    width = 0.34
    ax.bar(
        x - width/2,
        size_table["baseline_pass75_pct"],
        width,
        label="S2A-Net",
        color=C1,
    )
    ax.bar(
        x + width/2,
        size_table["refined_pass75_pct"],
        width,
        label="Refined",
        color=C2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(size_table["size_bin"])
    ax.set_xlabel("Object size, sqrt(rectangle area) [px]")
    ax.set_ylabel("Conditional pass@IoU 0.75 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_vector(fig, figures / "fig_size_bin_pass75")

    # ------------------------------------------------------------
    # TABLE 5 - Computational characterization
    # ------------------------------------------------------------
    # Use validated static complexity and measured runtime.
    computational = pd.DataFrame([
        {
            "component": "S2A-Net",
            "input": "1024x1024 patch",
            "parameters": 36398156,
            "GMAC": 171.776,
            "GFLOPs_2xMAC": 343.552,
            "checkpoint_bytes": 290635917,
            "mean_forward_ms": 17.061406007049776,
            "throughput_unit_per_s": 1000.0 / 17.061406007049776,
            "throughput_unit": "patch/s",
            "peak_cuda_allocated_MiB": 371.0673828125,
        },
        {
            "component": "YOLO26l-OBB",
            "input": "1024x1024 patch",
            "parameters": 27875556,
            "GMAC": np.nan,
            "GFLOPs_2xMAC": 259.701604352,
            "checkpoint_bytes": 168182477,
            "mean_forward_ms": 7.80244785428144,
            "throughput_unit_per_s": 128.16490653651337,
            "throughput_unit": "patch/s",
            "peak_cuda_allocated_MiB": 366.7841796875,
        },
        {
            "component": "ResNet-18 angle refiner",
            "input": "224x224 detection crop",
            "parameters": 11177538,
            "GMAC": 1.823522816,
            "GFLOPs_2xMAC": 3.647045632,
            "checkpoint_bytes": 44791883,
            "mean_forward_ms": 0.6742319416999817,
            "throughput_unit_per_s": 1483.1691264561564,
            "throughput_unit": "detection/s, batch=1",
            "peak_cuda_allocated_MiB": 185.81787109375,
        },
    ])
    computational.to_csv(tables / "table_computational_characterization.csv", index=False)

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    manifest = args.out_dir / "step6_manifest.txt"
    lines = [
        "=" * 96,
        "STEP 6 PUBLICATION-READY QUANTITATIVE OUTPUTS",
        "=" * 96,
        "",
        "FIGURES",
        "figures/fig_main_test_performance.pdf + .svg",
        "figures/fig_paired_bootstrap_ci.pdf + .svg",
        "figures/fig_angle_error_distribution.pdf + .svg",
        "figures/fig_iou_distribution.pdf + .svg",
        "figures/fig_size_bin_pass75.pdf + .svg",
        "",
        "SOURCE TABLES",
        "tables/table_main_test_metrics.csv",
        "tables/table_paired_bootstrap_ci.csv",
        "tables/table_geometry_summary.csv",
        "tables/table_size_bin_pass75.csv",
        "tables/table_computational_characterization.csv",
        "",
        "SCIENTIFIC NOTES",
        "- AP values are the previously validated frozen TEST values.",
        "- Bootstrap intervals come from the validated 5000-replicate paired original-image bootstrap.",
        "- Geometry plots use the 691 matched baseline TEST true positives from Step 3.",
        "- Conditional pass@0.75 is not AP75.",
        "- Refiner runtime is per detection crop; detector runtime is per 1024x1024 patch.",
        "- FLOP/MAC conventions differ by native tool; the computational table records the reported convention explicitly.",
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")

    print(manifest)
    print("DONE")


if __name__ == "__main__":
    main()
