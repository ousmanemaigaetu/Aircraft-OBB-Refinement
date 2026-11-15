#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the trained ResNet-18 residual OBB angle refiner on the external VAL split.

Scientific protocol
-------------------
- Uses the already selected checkpoint from the internal TRAIN/DEV split.
- Uses VAL only for external evaluation / later method selection.
- NEVER accesses TEST.
- Reuses the exact AngleDataset implementation from run_obb_angle_refiner.py
  so crop extraction and normalization remain consistent with training.
- Refines ANGLE ONLY.
- Preserves S2A-Net center, dimensions and confidence score.
- First evaluation is intentionally UNBOUNDED: no clamp, no gating.
- Saves raw/refined predictions, per-detection predictions, matched diagnostics,
  aspect-ratio analyses, timing, environment and figures.

The internal evaluator also checks that the untouched S2A-Net VAL predictions
reproduce the known baseline. If the baseline differs materially, refined AP
numbers must not be interpreted until the evaluator discrepancy is resolved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE_DARK = "#0649B8"
BLUE_MID = "#337EDA"
BLUE_LIGHT = "#8DBAF2"

KNOWN_BASELINE = {
    "AP50": 94.1715,
    "AP75": 56.6924,
    "mAP50_95": 55.0802,
}

ASPECT_BINS = [
    ("[1.00,1.10)", 1.00, 1.10),
    ("[1.10,1.20)", 1.10, 1.20),
    ("[1.20,1.30)", 1.20, 1.30),
    ("[1.30,1.50)", 1.30, 1.50),
    ("[1.50,2.00)", 1.50, 2.00),
    (">=2.00", 2.00, float("inf")),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="External VAL evaluation of the learned S2A-Net residual angle refiner."
    )
    p.add_argument(
        "--train-script",
        default="run_obb_angle_refiner.py",
        help="Training script whose AngleDataset/build_model are reused.",
    )
    p.add_argument(
        "--checkpoint",
        default="runs/angle_refiner/resnet18_angle/best_angle_refiner.pt",
    )
    p.add_argument(
        "--s2-val-pred",
        default="modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/val_eval/epoch_11/results_after_nms/plane.txt",
    )
    p.add_argument("--images-dir", default="dataset_avions/val/images")
    p.add_argument("--gt-dir", default="dataset_avions/val/annotations")
    p.add_argument(
        "--out-dir",
        default="runs/angle_refiner/resnet18_angle/val_eval_unbounded",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--fixed-score-thr",
        type=float,
        default=0.2259,
        help="Known S2A-Net VAL reporting threshold, used only for fixed-threshold P/R/F1 comparison.",
    )
    p.add_argument(
        "--baseline-tolerance-pt",
        type=float,
        default=0.05,
        help="Maximum absolute difference in percentage points allowed for baseline evaluator validation.",
    )
    return p.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_path(path: str | Path, kind: str = "file") -> Path:
    p = Path(path)
    if kind == "file" and not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}")
    if kind == "dir" and not p.is_dir():
        raise FileNotFoundError(f"Missing directory: {p}")
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(obj: Any, path: Path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def import_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("angle_train_module", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("AngleDataset", "build_model", "read_matches"):
        if not hasattr(mod, name):
            raise RuntimeError(f"Training script does not expose required symbol: {name}")
    return mod


def canonical_geom_from_poly(poly: np.ndarray) -> Dict[str, float]:
    """Geometry from ordered OBB corners, canonicalized to major >= minor and angle in [0, 180)."""
    p = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    e01 = p[1] - p[0]
    e12 = p[2] - p[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))

    if l01 >= l12:
        major, minor = l01, l12
        vec = e01
    else:
        major, minor = l12, l01
        vec = e12

    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
    return {
        "cx": float(c[0]),
        "cy": float(c[1]),
        "major": float(major),
        "minor": float(minor),
        "angle_deg": float(angle),
        "aspect_ratio": float(major / max(minor, 1e-12)),
        "area": float(major * minor),
    }


def rect_poly(cx: float, cy: float, major: float, minor: float, angle_deg: float) -> np.ndarray:
    """Create the polygon while preserving center/major/minor and changing only canonical major-axis angle."""
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)], dtype=np.float64)
    v = np.array([-math.sin(t), math.cos(t)], dtype=np.float64)
    du = 0.5 * major * u
    dv = 0.5 * minor * v
    c = np.array([cx, cy], dtype=np.float64)
    # Clockwise/counterclockwise ordering does not affect polygon IoU.
    return np.stack([c - du - dv, c + du - dv, c + du + dv, c - du + dv], axis=0)


def signed_axis_delta_deg(source_deg: float, target_deg: float) -> float:
    """target-source wrapped to [-90, 90)."""
    return float(((target_deg - source_deg + 90.0) % 180.0) - 90.0)


def abs_axis_error_deg(a: float, b: float) -> float:
    return abs(signed_axis_delta_deg(a, b))


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 2)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    inter = max(float(inter), 0.0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def read_detection_file(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 10:
                raise ValueError(f"Unexpected detection line {line_no}: {s}")
            image_id = parts[0]
            score = float(parts[1])
            vals = list(map(float, parts[2:10]))
            poly = np.array(vals, dtype=np.float64).reshape(4, 2)
            geom = canonical_geom_from_poly(poly)
            rows.append(
                {
                    "image_id": image_id,
                    "score": score,
                    "poly": poly,
                    "geom": geom,
                    "original_line": s,
                }
            )
    return rows


def write_detection_file(rows: List[Dict[str, Any]], path: Path, use_refined: bool):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            poly = r["refined_poly"] if use_refined else r["poly"]
            coords = " ".join(f"{float(x):.6f}" for x in poly.reshape(-1))
            f.write(f'{r["image_id"]} {r["score"]:.10f} {coords}\n')


def parse_gt_dir(gt_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    gt = defaultdict(list)
    for p in sorted(gt_dir.glob("*.txt")):
        image_id = p.stem
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                # DOTA annotation: x1 y1 ... x4 y4 class difficult
                if len(parts) < 9:
                    continue
                try:
                    coords = list(map(float, parts[:8]))
                except ValueError:
                    continue
                cls = parts[8]
                if cls != "plane":
                    continue
                difficult = 0
                if len(parts) >= 10:
                    try:
                        difficult = int(float(parts[9]))
                    except Exception:
                        difficult = 0
                poly = np.array(coords, dtype=np.float64).reshape(4, 2)
                gt[image_id].append(
                    {
                        "poly": poly,
                        "difficult": bool(difficult),
                        "geom": canonical_geom_from_poly(poly),
                    }
                )
    return dict(gt)


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_at_iou(
    detections: List[Dict[str, Any]],
    gt: Dict[str, List[Dict[str, Any]]],
    iou_thr: float,
    poly_key: str,
) -> Dict[str, Any]:
    npos = sum(1 for arr in gt.values() for g in arr if not g["difficult"])
    used = {img: np.zeros(len(arr), dtype=bool) for img, arr in gt.items()}

    ordered = sorted(enumerate(detections), key=lambda x: x[1]["score"], reverse=True)
    scores, tp, fp, ignored = [], [], [], []

    for _, det in ordered:
        image_id = det["image_id"]
        dpoly = det[poly_key]
        arr = gt.get(image_id, [])
        scores.append(float(det["score"]))

        if not arr:
            tp.append(0.0); fp.append(1.0); ignored.append(False)
            continue

        ious = np.array([polygon_iou(dpoly, g["poly"]) for g in arr], dtype=np.float64)
        j = int(np.argmax(ious))
        ov = float(ious[j])

        if ov >= iou_thr:
            if arr[j]["difficult"]:
                tp.append(0.0); fp.append(0.0); ignored.append(True)
            elif not used[image_id][j]:
                used[image_id][j] = True
                tp.append(1.0); fp.append(0.0); ignored.append(False)
            else:
                tp.append(0.0); fp.append(1.0); ignored.append(False)
        else:
            tp.append(0.0); fp.append(1.0); ignored.append(False)

    tp = np.array(tp, dtype=np.float64)
    fp = np.array(fp, dtype=np.float64)
    ignored = np.array(ignored, dtype=bool)
    scores = np.array(scores, dtype=np.float64)

    keep = ~ignored
    tp2 = tp[keep]
    fp2 = fp[keep]
    scores2 = scores[keep]
    ctp = np.cumsum(tp2)
    cfp = np.cumsum(fp2)
    rec = ctp / max(npos, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    ap = voc_ap(rec, prec)

    return {
        "ap": ap,
        "npos": int(npos),
        "scores": scores2,
        "tp": tp2,
        "fp": fp2,
        "rec": rec,
        "prec": prec,
    }


def summarize_detection_metrics(
    detections: List[Dict[str, Any]],
    gt: Dict[str, List[Dict[str, Any]]],
    poly_key: str,
    fixed_score_thr: float,
) -> Dict[str, Any]:
    thresholds = np.arange(0.50, 0.951, 0.05)
    per_thr = {}
    for thr in thresholds:
        r = evaluate_at_iou(detections, gt, float(round(thr, 2)), poly_key)
        per_thr[f"{thr:.2f}"] = r["ap"] * 100.0

    ap50 = per_thr["0.50"]
    ap75 = per_thr["0.75"]
    map5095 = float(np.mean(list(per_thr.values())))

    r50 = evaluate_at_iou(detections, gt, 0.50, poly_key)
    scores = r50["scores"]
    tp = r50["tp"]
    fp = r50["fp"]
    npos = r50["npos"]

    # Fixed score threshold
    mask = scores >= fixed_score_thr
    tp_fixed = int(tp[mask].sum())
    fp_fixed = int(fp[mask].sum())
    fn_fixed = int(max(npos - tp_fixed, 0))
    p_fixed = tp_fixed / max(tp_fixed + fp_fixed, 1)
    r_fixed = tp_fixed / max(npos, 1)
    f1_fixed = 2 * p_fixed * r_fixed / max(p_fixed + r_fixed, 1e-12)

    # Best F1 threshold on VAL, for reporting/tuning only.
    if len(scores):
        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        rec = ctp / max(npos, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-12)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        j = int(np.argmax(f1))
        best = {
            "score_threshold": float(scores[j]),
            "precision": float(prec[j] * 100.0),
            "recall": float(rec[j] * 100.0),
            "f1": float(f1[j] * 100.0),
            "tp": int(ctp[j]),
            "fp": int(cfp[j]),
            "fn": int(max(npos - ctp[j], 0)),
        }
    else:
        best = {}

    return {
        "AP50": float(ap50),
        "AP75": float(ap75),
        "mAP50_95": float(map5095),
        "AP_by_IoU": {k: float(v) for k, v in per_thr.items()},
        "fixed_score_threshold": {
            "threshold": float(fixed_score_thr),
            "precision": float(p_fixed * 100.0),
            "recall": float(r_fixed * 100.0),
            "f1": float(f1_fixed * 100.0),
            "tp": tp_fixed,
            "fp": fp_fixed,
            "fn": fn_fixed,
        },
        "best_f1_on_val": best,
        "num_positive_gt": int(npos),
    }


def build_val_rows_for_training_dataset(
    detections: List[Dict[str, Any]],
    images_dir: Path,
    columns: List[str],
) -> pd.DataFrame:
    rows = []
    by_img_rank = defaultdict(int)
    for det in detections:
        g = det["geom"]
        rank = by_img_rank[det["image_id"]]
        by_img_rank[det["image_id"]] += 1

        d = {c: 0 for c in columns}
        d.update(
            {
                "image_id": det["image_id"],
                "image_path": str((images_dir / f'{det["image_id"]}.png').resolve()),
                "det_rank": rank,
                "score": det["score"],
                "iou": 1.0,
                "s2_cx": g["cx"],
                "s2_cy": g["cy"],
                "s2_major": g["major"],
                "s2_minor": g["minor"],
                "s2_angle_deg": g["angle_deg"],
                "s2_aspect_ratio": g["aspect_ratio"],
                "s2_area": g["area"],
                # Dummy GT = S2. The target is never used during inference.
                "gt_cx": g["cx"],
                "gt_cy": g["cy"],
                "gt_major": g["major"],
                "gt_minor": g["minor"],
                "gt_angle_deg": g["angle_deg"],
                "gt_aspect_ratio": g["aspect_ratio"],
                "gt_area": g["area"],
                "delta_angle_deg": 0.0,
                "abs_delta_angle_deg": 0.0,
                "center_error_norm": 0.0,
                "delta_major_rel": 0.0,
                "delta_minor_rel": 0.0,
            }
        )
        p = det["poly"].reshape(-1)
        for i in range(4):
            d[f"pred_x{i+1}"] = float(p[2*i])
            d[f"pred_y{i+1}"] = float(p[2*i+1])
            d[f"gt_x{i+1}"] = float(p[2*i])
            d[f"gt_y{i+1}"] = float(p[2*i+1])
        rows.append(d)
    return pd.DataFrame(rows, columns=columns)


def instantiate_exact_dataset(train_mod, rows, args, ckpt_cfg):
    cls = train_mod.AngleDataset
    sig = inspect.signature(cls)
    kwargs = {}
    for name, param in sig.parameters.items():
        lname = name.lower()
        if lname in ("rows", "samples", "data", "matches"):
            kwargs[name] = rows
        elif lname in ("images_dir", "image_dir", "img_dir"):
            kwargs[name] = str(args.images_dir)
        elif lname in ("img_size", "image_size", "input_size", "out_size"):
            kwargs[name] = int(ckpt_cfg.get("img_size", 224))
        elif lname in ("crop_scale",):
            kwargs[name] = float(ckpt_cfg.get("crop_scale", 2.0))
        elif lname in ("min_crop_side", "min_side"):
            kwargs[name] = float(ckpt_cfg.get("min_crop_side", 64.0))
        elif lname in ("augment", "augmentation", "do_augment", "training", "train", "is_train"):
            kwargs[name] = False
        elif param.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(
                f"Cannot map required AngleDataset parameter '{name}'. "
                f"Constructor signature: {sig}"
            )
    ds = cls(**kwargs)
    return ds, str(sig), kwargs


def extract_input_tensor(sample: Any) -> torch.Tensor:
    """Find the 3xHxW input tensor in the exact training dataset sample."""
    if torch.is_tensor(sample):
        if sample.ndim == 3 and sample.shape[0] in (1, 3, 4):
            return sample
        raise RuntimeError(f"Unexpected tensor sample shape: {tuple(sample.shape)}")
    if isinstance(sample, (tuple, list)):
        for x in sample:
            try:
                return extract_input_tensor(x)
            except Exception:
                pass
    if isinstance(sample, dict):
        preferred = ("image", "img", "input", "x")
        for k in preferred:
            if k in sample:
                try:
                    return extract_input_tensor(sample[k])
                except Exception:
                    pass
        for x in sample.values():
            try:
                return extract_input_tensor(x)
            except Exception:
                pass
    raise RuntimeError(f"Could not locate image tensor in AngleDataset sample type {type(sample)}")


def build_exact_model(train_mod, ckpt):
    # Prefer the exact training factory.
    fn = train_mod.build_model
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        lname = name.lower()
        if lname == "pretrained":
            kwargs[name] = False
        elif param.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(f"Cannot map required build_model parameter '{name}', signature={sig}")
    obj = fn(**kwargs)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, (tuple, list)):
        models = [x for x in obj if isinstance(x, torch.nn.Module)]
        if not models:
            raise RuntimeError(f"build_model returned no torch.nn.Module: {type(obj)}")
        model = models[0]
    else:
        raise RuntimeError(f"Unexpected build_model return type: {type(obj)}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, str(sig)


def decode_vectors(raw: torch.Tensor) -> torch.Tensor:
    v = torch.nn.functional.normalize(raw, dim=1)
    return 0.5 * torch.atan2(v[:, 1], v[:, 0]) * (180.0 / math.pi)


def match_baseline_tp_for_diagnostics(
    detections: List[Dict[str, Any]],
    gt: Dict[str, List[Dict[str, Any]]],
    iou_thr: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Greedy score-ordered matching using baseline S2 geometry.
    Returned pairs are kept fixed when comparing refined geometry.
    """
    useful_indices = {
        img: [i for i, g in enumerate(arr) if not g["difficult"]]
        for img, arr in gt.items()
    }
    used = {img: set() for img in gt}
    pairs = []

    ordered = sorted(enumerate(detections), key=lambda x: x[1]["score"], reverse=True)
    for det_idx, det in ordered:
        arr = gt.get(det["image_id"], [])
        cand = useful_indices.get(det["image_id"], [])
        if not cand:
            continue
        vals = [(i, polygon_iou(det["poly"], arr[i]["poly"])) for i in cand if i not in used[det["image_id"]]]
        if not vals:
            continue
        j, iou = max(vals, key=lambda x: x[1])
        if iou >= iou_thr:
            used[det["image_id"]].add(j)
            pairs.append(
                {
                    "det_idx": det_idx,
                    "image_id": det["image_id"],
                    "gt_idx": j,
                    "baseline_iou": float(iou),
                }
            )
    return pairs


def stats(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "q90": float(np.quantile(a, 0.90)),
        "q95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def create_figures(diag: pd.DataFrame, det_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Angular error CDF
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for col, label, color in [
        ("angle_error_before_deg", "S2A-Net", BLUE_DARK),
        ("angle_error_after_deg", "S2A-Net + angle refiner", BLUE_MID),
    ]:
        x = np.sort(diag[col].dropna().to_numpy())
        if len(x):
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set_xlabel("Absolute angular error (deg)")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "angular_error_cdf.png", dpi=220)
    plt.close(fig)

    # IoU CDF
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for col, label, color in [
        ("baseline_iou", "S2A-Net", BLUE_DARK),
        ("refined_iou", "S2A-Net + angle refiner", BLUE_MID),
    ]:
        x = np.sort(diag[col].dropna().to_numpy())
        if len(x):
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "iou_cdf.png", dpi=220)
    plt.close(fig)

    # Predicted residual histogram
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(det_df["pred_delta_angle_deg"].to_numpy(), bins=60, color=BLUE_MID, alpha=0.9)
    ax.set_xlabel("Predicted residual angle (deg)")
    ax.set_ylabel("Detections")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_delta_hist.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)

    project_root = Path.cwd().resolve()
    train_script = ensure_path(args.train_script, "file")
    checkpoint_path = ensure_path(args.checkpoint, "file")
    s2_path = ensure_path(args.s2_val_pred, "file")
    images_dir = ensure_path(args.images_dir, "dir")
    gt_dir = ensure_path(args.gt_dir, "dir")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    # Save command/config immediately.
    config = vars(args).copy()
    config["project_root"] = str(project_root)
    json_dump(config, out / "config.json")

    print("=" * 90)
    print("EXTERNAL VAL - S2A-NET + RESIDUAL ANGLE REFINER")
    print("=" * 90)
    print("TEST ACCESS      : FORBIDDEN / NOT USED")
    print("Refinement       : angle only")
    print("Clamp / gating   : NONE (unbounded first evaluation)")
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"S2 VAL preds     : {s2_path}")
    print(f"VAL images       : {images_dir}")
    print(f"VAL annotations  : {gt_dir}")
    print(f"Output           : {out}")
    print("=" * 90)

    train_mod = import_training_module(train_script)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError("Unexpected checkpoint format.")
    ckpt_cfg = dict(ckpt.get("config", {}))

    detections = read_detection_file(s2_path)
    gt = parse_gt_dir(gt_dir)

    print(f"Detections loaded: {len(detections)}")
    print(f"GT images loaded : {len(gt)}")
    print(f"Checkpoint epoch : {ckpt.get('epoch')}")
    print(f"Checkpoint DEV median error: {ckpt.get('best_dev_median_abs_error_deg')} deg")

    # Save exact baseline file copy.
    shutil.copy2(s2_path, out / "baseline_plane_original.txt")

    # Build VAL rows using exact columns from TRAIN matches.
    train_matches_path = Path(ckpt_cfg.get("matches", "runs/angle_refiner/train_matches.csv"))
    if not train_matches_path.is_file():
        train_matches_path = Path("runs/angle_refiner/train_matches.csv")
    train_cols = pd.read_csv(train_matches_path, nrows=0).columns.tolist()
    val_df = build_val_rows_for_training_dataset(detections, images_dir, train_cols)
    val_rows_csv = out / "val_inference_rows.csv"
    val_df.to_csv(val_rows_csv, index=False)

    # Exact training parser + exact AngleDataset.
    rows = train_mod.read_matches(str(val_rows_csv))
    # Expose resolved images_dir to the dynamic constructor.
    args.images_dir = images_dir
    ds, ds_signature, ds_kwargs = instantiate_exact_dataset(train_mod, rows, args, ckpt_cfg)
    print(f"AngleDataset signature : {ds_signature}")
    print(f"AngleDataset length    : {len(ds)}")

    # Verify first exact sample before inference.
    sample0 = ds[0]
    x0 = extract_input_tensor(sample0)
    if tuple(x0.shape) != (3, int(ckpt_cfg.get("img_size", 224)), int(ckpt_cfg.get("img_size", 224))):
        raise RuntimeError(f"Unexpected exact-preprocessing tensor shape: {tuple(x0.shape)}")
    print(f"Exact input tensor     : {tuple(x0.shape)}")

    model, model_signature = build_exact_model(train_mod, ckpt)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Model parameters       : {total_params:,}")
    print(f"Trainable parameters   : {trainable_params:,}")
    print(f"Device                 : {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    pred_deltas = np.zeros(len(detections), dtype=np.float64)
    per_det_ms = np.zeros(len(detections), dtype=np.float64)

    wall_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ds), args.batch_size):
            stop = min(start + args.batch_size, len(ds))
            batch_cpu = []
            prep_t0 = time.perf_counter()
            for i in range(start, stop):
                xi = extract_input_tensor(ds[i]).float()
                batch_cpu.append(xi)
            x = torch.stack(batch_cpu, dim=0)
            prep_elapsed = time.perf_counter() - prep_t0

            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            raw = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_elapsed = time.perf_counter() - t0

            delta = decode_vectors(raw).detach().cpu().numpy()
            pred_deltas[start:stop] = delta
            # Per-detection total overhead = exact crop/preproc + model forward.
            ms = 1000.0 * (prep_elapsed + infer_elapsed) / max(stop - start, 1)
            per_det_ms[start:stop] = ms

            if start == 0 or stop == len(ds) or ((start // args.batch_size) + 1) % 10 == 0:
                print(f"VAL refiner {stop:4d}/{len(ds)} | batch avg total={ms:.3f} ms/det")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_wall = time.perf_counter() - wall_start

    # Apply angle only, no clamp/gating.
    per_detection_rows = []
    for i, (det, delta) in enumerate(zip(detections, pred_deltas)):
        g = det["geom"]
        new_angle = (g["angle_deg"] + float(delta)) % 180.0
        rpoly = rect_poly(g["cx"], g["cy"], g["major"], g["minor"], new_angle)
        det["pred_delta_angle_deg"] = float(delta)
        det["refined_angle_deg"] = float(new_angle)
        det["refined_poly"] = rpoly

        row = {
            "det_index": i,
            "image_id": det["image_id"],
            "score": det["score"],
            "s2_cx": g["cx"],
            "s2_cy": g["cy"],
            "s2_major": g["major"],
            "s2_minor": g["minor"],
            "s2_aspect_ratio": g["aspect_ratio"],
            "s2_angle_deg": g["angle_deg"],
            "pred_delta_angle_deg": float(delta),
            "refined_angle_deg": float(new_angle),
            "total_overhead_ms_per_det_batchavg": float(per_det_ms[i]),
        }
        for j in range(4):
            row[f"baseline_x{j+1}"] = float(det["poly"][j, 0])
            row[f"baseline_y{j+1}"] = float(det["poly"][j, 1])
            row[f"refined_x{j+1}"] = float(rpoly[j, 0])
            row[f"refined_y{j+1}"] = float(rpoly[j, 1])
        per_detection_rows.append(row)

    det_df = pd.DataFrame(per_detection_rows)
    det_df.to_csv(out / "per_detection_predictions.csv", index=False)

    write_detection_file(detections, out / "baseline_plane_reformatted.txt", use_refined=False)
    write_detection_file(detections, out / "refined_plane_unbounded.txt", use_refined=True)

    # AP evaluation.
    print("\nEvaluating untouched baseline with internal standalone-equivalent evaluator...")
    baseline_metrics = summarize_detection_metrics(
        detections, gt, "poly", args.fixed_score_thr
    )
    refined_metrics = summarize_detection_metrics(
        detections, gt, "refined_poly", args.fixed_score_thr
    )

    baseline_diff = {
        k: float(baseline_metrics[k] - KNOWN_BASELINE[k])
        for k in KNOWN_BASELINE
    }
    baseline_pass = all(abs(v) <= args.baseline_tolerance_pt for v in baseline_diff.values())

    metrics = {
        "baseline_internal_evaluator": baseline_metrics,
        "refined_unbounded_internal_evaluator": refined_metrics,
        "known_baseline_reference": KNOWN_BASELINE,
        "baseline_evaluator_difference_points": baseline_diff,
        "baseline_validation_tolerance_points": args.baseline_tolerance_pt,
        "baseline_evaluator_validation_passed": bool(baseline_pass),
        "delta_refined_minus_baseline_points": {
            "AP50": refined_metrics["AP50"] - baseline_metrics["AP50"],
            "AP75": refined_metrics["AP75"] - baseline_metrics["AP75"],
            "mAP50_95": refined_metrics["mAP50_95"] - baseline_metrics["mAP50_95"],
        },
    }
    json_dump(metrics, out / "metrics.json")

    # Fixed baseline TP pairs for geometric diagnostics.
    pairs = match_baseline_tp_for_diagnostics(detections, gt, 0.50)
    diag_rows = []
    for p in pairs:
        det = detections[p["det_idx"]]
        gto = gt[p["image_id"]][p["gt_idx"]]
        bg = det["geom"]
        gg = gto["geom"]
        refined_iou = polygon_iou(det["refined_poly"], gto["poly"])
        before_ang = abs_axis_error_deg(bg["angle_deg"], gg["angle_deg"])
        after_ang = abs_axis_error_deg(det["refined_angle_deg"], gg["angle_deg"])
        diag_rows.append(
            {
                "image_id": p["image_id"],
                "det_idx": p["det_idx"],
                "gt_idx": p["gt_idx"],
                "score": det["score"],
                "gt_aspect_ratio": gg["aspect_ratio"],
                "s2_aspect_ratio": bg["aspect_ratio"],
                "s2_angle_deg": bg["angle_deg"],
                "gt_angle_deg": gg["angle_deg"],
                "pred_delta_angle_deg": det["pred_delta_angle_deg"],
                "refined_angle_deg": det["refined_angle_deg"],
                "angle_error_before_deg": before_ang,
                "angle_error_after_deg": after_ang,
                "angle_error_gain_deg": before_ang - after_ang,
                "baseline_iou": p["baseline_iou"],
                "refined_iou": refined_iou,
                "iou_gain": refined_iou - p["baseline_iou"],
                "baseline_pass75": int(p["baseline_iou"] >= 0.75),
                "refined_pass75": int(refined_iou >= 0.75),
            }
        )

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out / "matched_diagnostics_baseline_tp.csv", index=False)

    # Summary diagnostics.
    angular_summary = {
        "baseline_angle_error_deg": stats(diag["angle_error_before_deg"]),
        "refined_angle_error_deg": stats(diag["angle_error_after_deg"]),
        "angle_error_gain_deg": stats(diag["angle_error_gain_deg"]),
        "predicted_delta_all_detections_deg": stats(pred_deltas),
    }
    iou_summary = {
        "baseline_iou": stats(diag["baseline_iou"]),
        "refined_iou": stats(diag["refined_iou"]),
        "iou_gain": stats(diag["iou_gain"]),
        "baseline_pass75_percent": float(100.0 * diag["baseline_pass75"].mean()) if len(diag) else None,
        "refined_pass75_percent": float(100.0 * diag["refined_pass75"].mean()) if len(diag) else None,
        "pass75_gain_points": float(100.0 * (diag["refined_pass75"].mean() - diag["baseline_pass75"].mean())) if len(diag) else None,
        "improved_iou_gt_0p01_percent": float(100.0 * (diag["iou_gain"] > 0.01).mean()) if len(diag) else None,
        "degraded_iou_lt_minus0p01_percent": float(100.0 * (diag["iou_gain"] < -0.01).mean()) if len(diag) else None,
    }
    json_dump(angular_summary, out / "angular_error_summary.json")
    json_dump(iou_summary, out / "iou_summary.json")

    # Aspect-ratio diagnostics.
    aspect_rows = []
    for label, lo, hi in ASPECT_BINS:
        m = (diag["gt_aspect_ratio"] >= lo) & (diag["gt_aspect_ratio"] < hi)
        d = diag[m]
        if len(d) == 0:
            continue
        aspect_rows.append(
            {
                "aspect_bin": label,
                "n": len(d),
                "median_angle_before_deg": d["angle_error_before_deg"].median(),
                "median_angle_after_deg": d["angle_error_after_deg"].median(),
                "median_angle_gain_deg": d["angle_error_gain_deg"].median(),
                "q90_angle_before_deg": d["angle_error_before_deg"].quantile(0.90),
                "q90_angle_after_deg": d["angle_error_after_deg"].quantile(0.90),
                "median_iou_before": d["baseline_iou"].median(),
                "median_iou_after": d["refined_iou"].median(),
                "median_iou_gain": d["iou_gain"].median(),
                "pass75_before_percent": 100.0 * d["baseline_pass75"].mean(),
                "pass75_after_percent": 100.0 * d["refined_pass75"].mean(),
                "pass75_gain_points": 100.0 * (d["refined_pass75"].mean() - d["baseline_pass75"].mean()),
                "improved_iou_gt_0p01_percent": 100.0 * (d["iou_gain"] > 0.01).mean(),
                "degraded_iou_lt_minus0p01_percent": 100.0 * (d["iou_gain"] < -0.01).mean(),
            }
        )
    pd.DataFrame(aspect_rows).to_csv(out / "aspect_ratio_diagnostics.csv", index=False)

    # Environment + timing.
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = None

    env = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    }
    json_dump(env, out / "environment.json")

    timing = {
        "num_detections": len(detections),
        "total_wall_seconds_crop_preprocess_plus_refiner": total_wall,
        "mean_total_overhead_ms_per_detection": float(np.mean(per_det_ms)),
        "median_batchavg_overhead_ms_per_detection": float(np.median(per_det_ms)),
        "approx_total_overhead_ms_per_original_image": float(1000.0 * total_wall / max(len(list(images_dir.glob("*"))), 1)),
        "peak_cuda_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else None,
    }
    json_dump(timing, out / "timing.json")

    run_summary = {
        "split": "VAL",
        "test_used": False,
        "refinement": "angle_only_unbounded",
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_best_dev_median_abs_error_deg": float(ckpt.get("best_dev_median_abs_error_deg")),
        "num_predictions": len(detections),
        "num_baseline_tp_pairs_for_geometry": len(diag),
        "model_total_parameters": total_params,
        "model_trainable_parameters": trainable_params,
        "angle_dataset_signature": ds_signature,
        "build_model_signature": model_signature,
        "baseline_evaluator_validation_passed": baseline_pass,
        "baseline_metrics": baseline_metrics,
        "refined_metrics": refined_metrics,
        "metric_deltas_points": metrics["delta_refined_minus_baseline_points"],
        "angular_summary": angular_summary,
        "iou_summary": iou_summary,
        "timing": timing,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_script_sha256": sha256_file(train_script),
    }
    json_dump(run_summary, out / "run_summary.json")

    # Reproducibility snapshots.
    shutil.copy2(train_script, out / "training_script_snapshot.py")
    shutil.copy2(checkpoint_path, out / "best_angle_refiner_snapshot.pt")

    create_figures(diag, det_df, out / "figures")

    print("\n" + "=" * 90)
    print("RESULTATS VAL - PREMIERE EVALUATION NON BORNEE")
    print("=" * 90)
    print("Baseline evaluator validation :", "PASS" if baseline_pass else "FAIL")
    print(
        f"Baseline internal : AP50={baseline_metrics['AP50']:.4f} | "
        f"AP75={baseline_metrics['AP75']:.4f} | mAP={baseline_metrics['mAP50_95']:.4f}"
    )
    print(
        f"Refined unbounded : AP50={refined_metrics['AP50']:.4f} | "
        f"AP75={refined_metrics['AP75']:.4f} | mAP={refined_metrics['mAP50_95']:.4f}"
    )
    d = metrics["delta_refined_minus_baseline_points"]
    print(
        f"Delta points      : AP50={d['AP50']:+.4f} | "
        f"AP75={d['AP75']:+.4f} | mAP={d['mAP50_95']:+.4f}"
    )
    print(
        f"Angle median matched TP: "
        f"{angular_summary['baseline_angle_error_deg']['median']:.4f} -> "
        f"{angular_summary['refined_angle_error_deg']['median']:.4f} deg"
    )
    print(
        f"IoU median matched TP  : "
        f"{iou_summary['baseline_iou']['median']:.4f} -> "
        f"{iou_summary['refined_iou']['median']:.4f}"
    )
    print(
        f"Pass@0.75 matched TP   : "
        f"{iou_summary['baseline_pass75_percent']:.2f}% -> "
        f"{iou_summary['refined_pass75_percent']:.2f}% "
        f"({iou_summary['pass75_gain_points']:+.2f} pts)"
    )
    print(f"Model params           : {total_params:,}")
    print(f"Mean overhead          : {timing['mean_total_overhead_ms_per_detection']:.3f} ms/det")
    print(f"Output                  : {out}")
    print("TEST USED               : False")
    print("=" * 90)

    if not baseline_pass:
        print("\nWARNING:")
        print("The internal evaluator did NOT reproduce the known S2A-Net VAL baseline within tolerance.")
        print("Do not interpret refined AP metrics yet. The refined plane.txt and all diagnostics are saved,")
        print("so the existing validated standalone evaluator can be used on refined_plane_unbounded.txt.")


if __name__ == "__main__":
    main()
