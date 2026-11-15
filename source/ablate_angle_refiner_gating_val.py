#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Targeted VAL ablation for the learned residual OBB angle refiner.

Uses ONLY already-computed VAL residual predictions.
No neural-network inference is rerun.
TEST is never accessed.

Variants:
- unbounded
- gate_abs_gt_30
- gate_abs_gt_45
- gate_abs_gt_60
- clamp_abs_60

Each generated prediction file is evaluated with the already validated
eval_obb_val_standalone.py evaluator. All stdout and metrics are saved.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pred-csv",
        default="runs/angle_refiner/resnet18_angle/val_eval_unbounded/per_detection_predictions.csv",
    )
    p.add_argument(
        "--baseline-pred",
        default="modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/val_eval/epoch_11/results_after_nms/plane.txt",
    )
    p.add_argument("--gt-dir", default="dataset_avions/val/annotations")
    p.add_argument("--evaluator", default="eval_obb_val_standalone.py")
    p.add_argument(
        "--out-dir",
        default="runs/angle_refiner/resnet18_angle/val_gating_ablation",
    )
    p.add_argument("--score-thr", type=float, default=0.2259)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rect_poly(cx, cy, major, minor, angle_deg):
    t = math.radians(float(angle_deg))
    u = np.array([math.cos(t), math.sin(t)], dtype=np.float64)
    v = np.array([-math.sin(t), math.cos(t)], dtype=np.float64)
    c = np.array([float(cx), float(cy)], dtype=np.float64)
    du = 0.5 * float(major) * u
    dv = 0.5 * float(minor) * v
    return np.stack(
        [c - du - dv, c + du - dv, c + du + dv, c - du + dv], axis=0
    )


def write_plane(df, path, mode, threshold=None):
    num_modified = 0
    num_gated = 0
    num_clipped = 0
    deltas_used = []

    with path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            delta = float(r["pred_delta_angle_deg"])
            used = delta

            if mode == "unbounded":
                pass
            elif mode == "gate":
                if abs(delta) > float(threshold):
                    used = 0.0
                    num_gated += 1
            elif mode == "clamp":
                clipped = float(np.clip(delta, -float(threshold), float(threshold)))
                if clipped != delta:
                    num_clipped += 1
                used = clipped
            else:
                raise ValueError(mode)

            if abs(used) > 1e-12:
                num_modified += 1

            angle = (float(r["s2_angle_deg"]) + used) % 180.0
            poly = rect_poly(
                r["s2_cx"], r["s2_cy"],
                r["s2_major"], r["s2_minor"],
                angle
            )
            coords = " ".join(f"{x:.6f}" for x in poly.reshape(-1))
            f.write(f'{r["image_id"]} {float(r["score"]):.10f} {coords}\n')
            deltas_used.append(used)

    return {
        "num_predictions": int(len(df)),
        "num_modified": int(num_modified),
        "num_gated": int(num_gated),
        "num_clipped": int(num_clipped),
        "used_delta_abs_median_deg": float(np.median(np.abs(deltas_used))),
        "used_delta_abs_q90_deg": float(np.quantile(np.abs(deltas_used), 0.90)),
        "used_delta_abs_q95_deg": float(np.quantile(np.abs(deltas_used), 0.95)),
        "used_delta_abs_max_deg": float(np.max(np.abs(deltas_used))),
    }


def parse_evaluator_output(txt):
    marker = "YOLO26l-OBB"
    if marker not in txt:
        raise RuntimeError("Could not locate second-model section in evaluator output.")
    sec = txt.split(marker, 1)[1]
    if "COMPARAISON YOLO - S²A-Net" in sec:
        sec = sec.split("COMPARAISON YOLO - S²A-Net", 1)[0]

    def grab(pattern):
        m = re.search(pattern, sec)
        return float(m.group(1)) if m else None

    ap_by_iou = {}
    for thr, val in re.findall(r"AP@([0-9.]+):\s*([0-9.]+)\s*%", sec):
        ap_by_iou[str(thr)] = float(val)

    mcounts = re.search(r"TP / FP / FN\s*:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", sec)

    out = {
        "AP50": grab(r"AP50\s*:\s*([0-9.]+)\s*%"),
        "AP75": grab(r"AP75\s*:\s*([0-9.]+)\s*%"),
        "mAP50_95": grab(r"mAP50-95\s*:\s*([0-9.]+)\s*%"),
        "precision": grab(r"Precision\s*:\s*([0-9.]+)\s*%"),
        "recall": grab(r"Recall\s*:\s*([0-9.]+)\s*%"),
        "f1": grab(r"F1\s*:\s*([0-9.]+)\s*%"),
        "AP_by_IoU": ap_by_iou,
    }
    if mcounts:
        out["tp"] = int(mcounts.group(1))
        out["fp"] = int(mcounts.group(2))
        out["fn"] = int(mcounts.group(3))
    return out


def run_eval(evaluator, gt_dir, baseline, variant, score_thr):
    cmd = [
        sys.executable, str(evaluator),
        "--gt-dir", str(gt_dir),
        "--s2-pred", str(baseline),
        "--yolo-pred", str(variant),
        "--class-name", "plane",
        "--s2-score", str(score_thr),
        "--yolo-score", str(score_thr),
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    stdout = p.stdout + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(
            f"Evaluator failed for {variant} with code {p.returncode}\n{stdout}"
        )
    return cmd, stdout, parse_evaluator_output(stdout)


def main():
    args = parse_args()

    pred_csv = Path(args.pred_csv)
    baseline = Path(args.baseline_pred)
    gt_dir = Path(args.gt_dir)
    evaluator = Path(args.evaluator)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = out / "predictions"
    log_dir = out / "evaluator_logs"
    pred_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    for p in [pred_csv, baseline, evaluator]:
        if not p.is_file():
            raise FileNotFoundError(p)
    if not gt_dir.is_dir():
        raise FileNotFoundError(gt_dir)

    df = pd.read_csv(pred_csv)

    required = {
        "image_id", "score", "s2_cx", "s2_cy", "s2_major", "s2_minor",
        "s2_angle_deg", "pred_delta_angle_deg"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in prediction CSV: {sorted(missing)}")

    variants = [
        ("unbounded", "unbounded", None),
        ("gate_abs_gt_30", "gate", 30.0),
        ("gate_abs_gt_45", "gate", 45.0),
        ("gate_abs_gt_60", "gate", 60.0),
        ("clamp_abs_60", "clamp", 60.0),
    ]

    config = {
        "split": "VAL",
        "test_used": False,
        "pred_csv": str(pred_csv),
        "baseline_pred": str(baseline),
        "gt_dir": str(gt_dir),
        "evaluator": str(evaluator),
        "score_threshold": args.score_thr,
        "seed": args.seed,
        "variants": [
            {"name": n, "mode": m, "threshold_deg": t}
            for n, m, t in variants
        ],
        "scientific_reason": (
            "Targeted gating ablation motivated by VAL diagnostics: corrections "
            "from about 5 to 60 deg remained useful, while the >=60 deg bin "
            "showed near-zero mean IoU gain and increased degradation."
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []

    print("=" * 96)
    print("TARGETED VAL GATING ABLATION")
    print("=" * 96)
    print("TEST USED : False")
    print(f"Predictions: {len(df)}")
    print("=" * 96)

    for name, mode, thr in variants:
        pred_path = pred_dir / f"{name}.txt"
        stats = write_plane(df, pred_path, mode, thr)
        cmd, stdout, metrics = run_eval(
            evaluator, gt_dir, baseline, pred_path, args.score_thr
        )
        (log_dir / f"{name}.txt").write_text(stdout, encoding="utf-8")

        row = {
            "variant": name,
            "mode": mode,
            "threshold_deg": thr,
            **stats,
            "AP50": metrics["AP50"],
            "AP75": metrics["AP75"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
        }
        for k, v in metrics["AP_by_IoU"].items():
            row[f"AP@{k}"] = v
        rows.append(row)

        print(
            f"{name:16s} | "
            f"AP50={metrics['AP50']:.4f} | "
            f"AP75={metrics['AP75']:.4f} | "
            f"mAP={metrics['mAP50_95']:.4f} | "
            f"gated={stats['num_gated']:4d} | clipped={stats['num_clipped']:4d}"
        )

    result = pd.DataFrame(rows)

    ub = result[result["variant"] == "unbounded"].iloc[0]
    known_s2 = {"AP50": 94.1715, "AP75": 56.6924, "mAP50_95": 55.0802}
    for metric in ["AP50", "AP75", "mAP50_95"]:
        result[f"delta_vs_unbounded_{metric}"] = result[metric] - float(ub[metric])
        result[f"delta_vs_S2_{metric}"] = result[metric] - known_s2[metric]

    result.to_csv(out / "ablation_summary.csv", index=False)

    best = result.sort_values(
        ["AP75", "mAP50_95", "AP50"],
        ascending=[False, False, False]
    ).iloc[0].to_dict()

    summary = {
        "split": "VAL",
        "test_used": False,
        "selection_rule": "maximize AP75 on VAL; tie-break mAP50-95, then AP50",
        "best_variant": best,
        "known_s2_baseline": known_s2,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_torch": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 96)
    print("BEST VAL VARIANT")
    print("=" * 96)
    print(f"Variant : {best['variant']}")
    print(f"AP50    : {best['AP50']:.4f}")
    print(f"AP75    : {best['AP75']:.4f}")
    print(f"mAP     : {best['mAP50_95']:.4f}")
    print(f"Output  : {out}")
    print("TEST USED : False")
    print("=" * 96)


if __name__ == "__main__":
    main()
