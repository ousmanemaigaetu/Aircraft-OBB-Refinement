#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FINAL FROZEN TEST evaluation for the learned S2A-Net residual angle refiner.

FROZEN ON VAL BEFORE TEST:
- checkpoint: best_angle_refiner.pt, epoch 10
- architecture: ResNet-18
- crop scale: 2.0
- input size: 224
- minimum crop side: 64
- refinement: ANGLE ONLY
- inference rule: clip predicted delta-angle to [-60, +60] degrees
- center, dimensions and confidence score: preserved from S2A-Net

IMPORTANT:
- This script evaluates exactly ONE frozen refinement rule on TEST.
- It does NOT compute or compare alternative TEST variants.
- It reuses the exact training AngleDataset/build_model implementation.
- It saves all outputs required for reproducibility and the article.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

FROZEN_CLAMP_DEG = 60.0

FROZEN_VAL_SELECTION = {
    "baseline_s2": {
        "AP50": 94.1715,
        "AP75": 56.6924,
        "mAP50_95": 55.0802,
    },
    "unbounded": {
        "AP50": 94.1707,
        "AP75": 65.5807,
        "mAP50_95": 58.5562,
    },
    "clamp_abs_60": {
        "AP50": 94.1707,
        "AP75": 67.2749,
        "mAP50_95": 58.9133,
    },
    "selection_rule": "maximize AP75 on VAL; tie-break mAP50-95, then AP50",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-script", default="run_obb_angle_refiner.py")
    p.add_argument("--checkpoint", default="runs/angle_refiner/resnet18_angle/best_angle_refiner.pt")
    p.add_argument(
        "--s2-test-pred",
        default="modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/test_final/results_after_nms/plane.txt",
    )
    p.add_argument("--images-dir", default="dataset_avions/test/images")
    p.add_argument("--gt-dir", default="dataset_avions/test/annotations")
    p.add_argument("--evaluator", default="eval_obb_val_standalone.py")
    p.add_argument(
        "--out-dir",
        default="runs/angle_refiner/resnet18_angle/test_final_frozen_clamp60",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.50,
        help="Fixed TEST score threshold for P/R/F1 reporting only. AP is threshold-independent.",
    )
    return p.parse_args()


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def jdump(obj, path):
    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def canonical_geom(poly):
    p = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    e01 = p[1] - p[0]
    e12 = p[2] - p[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))
    if l01 >= l12:
        major, minor, vec = l01, l12, e01
    else:
        major, minor, vec = l12, l01, e12
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
    return {
        "cx": float(c[0]),
        "cy": float(c[1]),
        "major": major,
        "minor": minor,
        "angle_deg": float(angle),
        "aspect_ratio": float(major / max(minor, 1e-12)),
        "area": float(major * minor),
    }


def rect_poly(cx, cy, major, minor, angle_deg):
    t = math.radians(float(angle_deg))
    u = np.array([math.cos(t), math.sin(t)], dtype=np.float64)
    v = np.array([-math.sin(t), math.cos(t)], dtype=np.float64)
    c = np.array([float(cx), float(cy)], dtype=np.float64)
    du = 0.5 * float(major) * u
    dv = 0.5 * float(minor) * v
    return np.stack([c-du-dv, c+du-dv, c+du+dv, c-du+dv], axis=0)


def signed_axis_delta(source_deg, target_deg):
    return float(((target_deg - source_deg + 90.0) % 180.0) - 90.0)


def axis_error(a, b):
    return abs(signed_axis_delta(a, b))


def polygon_iou(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    aa = abs(float(cv2.contourArea(a)))
    ab = abs(float(cv2.contourArea(b)))
    if aa <= 0 or ab <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    inter = max(float(inter), 0.0)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def read_dets(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            q = s.split()
            if len(q) < 10:
                raise ValueError(f"Bad detection line: {s}")
            poly = np.array(list(map(float, q[2:10])), dtype=np.float64).reshape(4, 2)
            out.append({
                "image_id": q[0],
                "score": float(q[1]),
                "poly": poly,
                "geom": canonical_geom(poly),
            })
    return out


def parse_gt(gt_dir):
    out = defaultdict(list)
    for p in sorted(Path(gt_dir).glob("*.txt")):
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                q = line.strip().split()
                if len(q) < 9:
                    continue
                try:
                    coords = list(map(float, q[:8]))
                except Exception:
                    continue
                if q[8] != "plane":
                    continue
                difficult = False
                if len(q) >= 10:
                    try:
                        difficult = bool(int(float(q[9])))
                    except Exception:
                        difficult = False
                poly = np.array(coords, dtype=np.float64).reshape(4, 2)
                out[p.stem].append({
                    "poly": poly,
                    "difficult": difficult,
                    "geom": canonical_geom(poly),
                })
    return dict(out)


def build_inference_dataframe(dets, images_dir, columns):
    rows = []
    ranks = defaultdict(int)
    for d in dets:
        g = d["geom"]
        r = {c: 0 for c in columns}
        rank = ranks[d["image_id"]]
        ranks[d["image_id"]] += 1
        r.update({
            "image_id": d["image_id"],
            "image_path": str((Path(images_dir) / f'{d["image_id"]}.png').resolve()),
            "det_rank": rank,
            "score": d["score"],
            "iou": 1.0,
            "s2_cx": g["cx"], "s2_cy": g["cy"],
            "s2_major": g["major"], "s2_minor": g["minor"],
            "s2_angle_deg": g["angle_deg"],
            "s2_aspect_ratio": g["aspect_ratio"],
            "s2_area": g["area"],
            "gt_cx": g["cx"], "gt_cy": g["cy"],
            "gt_major": g["major"], "gt_minor": g["minor"],
            "gt_angle_deg": g["angle_deg"],
            "gt_aspect_ratio": g["aspect_ratio"],
            "gt_area": g["area"],
            "delta_angle_deg": 0.0,
            "abs_delta_angle_deg": 0.0,
            "center_error_norm": 0.0,
            "delta_major_rel": 0.0,
            "delta_minor_rel": 0.0,
        })
        flat = d["poly"].reshape(-1)
        for i in range(4):
            r[f"pred_x{i+1}"] = float(flat[2*i])
            r[f"pred_y{i+1}"] = float(flat[2*i+1])
            r[f"gt_x{i+1}"] = float(flat[2*i])
            r[f"gt_y{i+1}"] = float(flat[2*i+1])
        rows.append(r)
    return pd.DataFrame(rows, columns=columns)


def instantiate_dataset(train_mod, rows, images_dir, cfg):
    sig = inspect.signature(train_mod.AngleDataset)
    kwargs = {}
    for name, p in sig.parameters.items():
        n = name.lower()
        if n in ("rows", "samples", "data", "matches"):
            kwargs[name] = rows
        elif n in ("images_dir", "image_dir", "img_dir"):
            kwargs[name] = str(images_dir)
        elif n in ("img_size", "image_size", "input_size", "out_size"):
            kwargs[name] = int(cfg.get("img_size", 224))
        elif n == "crop_scale":
            kwargs[name] = float(cfg.get("crop_scale", 2.0))
        elif n in ("min_crop_side", "min_side"):
            kwargs[name] = float(cfg.get("min_crop_side", 64.0))
        elif n in ("augment", "augmentation", "do_augment", "training", "train", "is_train"):
            kwargs[name] = False
        elif p.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(f"Cannot map AngleDataset parameter {name}, signature={sig}")
    return train_mod.AngleDataset(**kwargs), str(sig)


def extract_image_tensor(sample):
    if torch.is_tensor(sample):
        if sample.ndim == 3:
            return sample
    if isinstance(sample, (tuple, list)):
        for x in sample:
            try:
                return extract_image_tensor(x)
            except Exception:
                pass
    if isinstance(sample, dict):
        for k in ("image", "img", "input", "x"):
            if k in sample:
                try:
                    return extract_image_tensor(sample[k])
                except Exception:
                    pass
        for x in sample.values():
            try:
                return extract_image_tensor(x)
            except Exception:
                pass
    raise RuntimeError(f"Cannot find image tensor in sample type {type(sample)}")


def build_model(train_mod, ckpt):
    sig = inspect.signature(train_mod.build_model)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name.lower() == "pretrained":
            kwargs[name] = False
        elif p.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(f"Cannot map build_model parameter {name}")
    obj = train_mod.build_model(**kwargs)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, (tuple, list)):
        model = next(x for x in obj if isinstance(x, torch.nn.Module))
    else:
        raise RuntimeError(f"Unexpected build_model return type {type(obj)}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


def decode(raw):
    v = torch.nn.functional.normalize(raw, dim=1)
    return 0.5 * torch.atan2(v[:, 1], v[:, 0]) * 180.0 / math.pi


def write_prediction_file(dets, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in dets:
            coords = " ".join(f"{x:.6f}" for x in d["refined_poly"].reshape(-1))
            f.write(f'{d["image_id"]} {d["score"]:.10f} {coords}\n')


def fixed_baseline_matches(dets, gt, thr=0.50):
    useful = {im: [i for i,g in enumerate(arr) if not g["difficult"]] for im,arr in gt.items()}
    used = {im:set() for im in gt}
    pairs = []
    for di,d in sorted(enumerate(dets), key=lambda x:x[1]["score"], reverse=True):
        arr = gt.get(d["image_id"], [])
        vals = [(j, polygon_iou(d["poly"], arr[j]["poly"]))
                for j in useful.get(d["image_id"], [])
                if j not in used[d["image_id"]]]
        if not vals:
            continue
        j,iou = max(vals, key=lambda z:z[1])
        if iou >= thr:
            used[d["image_id"]].add(j)
            pairs.append((di, d["image_id"], j, float(iou)))
    return pairs


def summary_stats(x):
    x = np.asarray(list(x), dtype=np.float64)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def run_validated_evaluator(evaluator, gt_dir, baseline, refined, score_thr):
    cmd = [
        sys.executable, str(evaluator),
        "--gt-dir", str(gt_dir),
        "--s2-pred", str(baseline),
        "--yolo-pred", str(refined),
        "--class-name", "plane",
        "--s2-score", str(score_thr),
        "--yolo-score", str(score_thr),
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    text = p.stdout + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(text)
    return cmd, text


def parse_second_section(text):
    if "YOLO26l-OBB" not in text:
        raise RuntimeError("Could not parse evaluator output.")
    sec = text.split("YOLO26l-OBB", 1)[1]
    if "COMPARAISON YOLO - S²A-Net" in sec:
        sec = sec.split("COMPARAISON YOLO - S²A-Net", 1)[0]

    import re
    def g(pat):
        m = re.search(pat, sec)
        return float(m.group(1)) if m else None

    aps = {thr: float(val) for thr,val in re.findall(r"AP@([0-9.]+):\s*([0-9.]+)\s*%", sec)}
    m = re.search(r"TP / FP / FN\s*:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", sec)
    return {
        "AP50": g(r"AP50\s*:\s*([0-9.]+)\s*%"),
        "AP75": g(r"AP75\s*:\s*([0-9.]+)\s*%"),
        "mAP50_95": g(r"mAP50-95\s*:\s*([0-9.]+)\s*%"),
        "precision": g(r"Precision\s*:\s*([0-9.]+)\s*%"),
        "recall": g(r"Recall\s*:\s*([0-9.]+)\s*%"),
        "f1": g(r"F1\s*:\s*([0-9.]+)\s*%"),
        "tp": int(m.group(1)) if m else None,
        "fp": int(m.group(2)) if m else None,
        "fn": int(m.group(3)) if m else None,
        "AP_by_IoU": aps,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    train_script = Path(args.train_script)
    ckpt_path = Path(args.checkpoint)
    s2_path = Path(args.s2_test_pred)
    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    evaluator = Path(args.evaluator)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p in (train_script, ckpt_path, s2_path, evaluator):
        if not p.is_file():
            raise FileNotFoundError(p)
    for p in (images_dir, gt_dir):
        if not p.is_dir():
            raise FileNotFoundError(p)

    frozen_config = {
        "split": "TEST",
        "test_used": True,
        "method_frozen_before_test": True,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": 10,
        "architecture": "ResNet-18",
        "pretraining": "ImageNet",
        "input_size": 224,
        "crop_scale": 2.0,
        "min_crop_side": 64.0,
        "refined_components": ["angle"],
        "preserved_components": ["center", "major", "minor", "score"],
        "delta_rule": "clip(predicted_delta_angle_deg, -60, +60)",
        "clamp_deg": FROZEN_CLAMP_DEG,
        "selection_source": "VAL only",
        "val_selection_results": FROZEN_VAL_SELECTION,
        "score_threshold_for_PRF1_only": args.score_thr,
        "seed": args.seed,
    }
    jdump(frozen_config, out / "frozen_config.json")

    print("=" * 92)
    print("FINAL FROZEN TEST - S2A-NET + RESIDUAL ANGLE REFINER")
    print("=" * 92)
    print("Method frozen before TEST : True")
    print("Checkpoint                : epoch 10")
    print("Refinement                : angle only")
    print("Frozen rule               : clip delta to [-60, +60] deg")
    print("No TEST variant search    : True")
    print("=" * 92)

    train_mod = import_module(train_script, "angle_train_module")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {}))

    dets = read_dets(s2_path)
    gt = parse_gt(gt_dir)

    matches_path = Path(cfg.get("matches", "runs/angle_refiner/train_matches.csv"))
    train_cols = pd.read_csv(matches_path, nrows=0).columns.tolist()
    infer_df = build_inference_dataframe(dets, images_dir, train_cols)
    infer_csv = out / "test_inference_rows.csv"
    infer_df.to_csv(infer_csv, index=False)

    rows = train_mod.read_matches(str(infer_csv))
    ds, ds_sig = instantiate_dataset(train_mod, rows, images_dir, cfg)

    model = build_model(train_mod, ckpt)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    pred_raw = np.zeros(len(dets), dtype=np.float64)
    pred_used = np.zeros(len(dets), dtype=np.float64)

    t_all = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ds), args.batch_size):
            stop = min(start + args.batch_size, len(ds))
            batch = torch.stack([extract_image_tensor(ds[i]).float() for i in range(start, stop)])
            batch = batch.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            raw = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            d = decode(raw).cpu().numpy()
            pred_raw[start:stop] = d
            pred_used[start:stop] = np.clip(d, -FROZEN_CLAMP_DEG, FROZEN_CLAMP_DEG)
            if start == 0 or stop == len(ds) or ((start // args.batch_size)+1) % 10 == 0:
                print(f"TEST refinement {stop}/{len(ds)}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - t_all

    per_rows = []
    for i,(det,raw_d,use_d) in enumerate(zip(dets,pred_raw,pred_used)):
        g = det["geom"]
        a = (g["angle_deg"] + float(use_d)) % 180.0
        det["raw_delta"] = float(raw_d)
        det["used_delta"] = float(use_d)
        det["refined_angle_deg"] = float(a)
        det["refined_poly"] = rect_poly(g["cx"], g["cy"], g["major"], g["minor"], a)
        per_rows.append({
            "det_index": i,
            "image_id": det["image_id"],
            "score": det["score"],
            "s2_cx": g["cx"], "s2_cy": g["cy"],
            "s2_major": g["major"], "s2_minor": g["minor"],
            "s2_aspect_ratio": g["aspect_ratio"],
            "s2_angle_deg": g["angle_deg"],
            "pred_delta_angle_raw_deg": float(raw_d),
            "pred_delta_angle_used_deg": float(use_d),
            "was_clipped": int(abs(raw_d) > FROZEN_CLAMP_DEG),
            "refined_angle_deg": float(a),
        })

    pd.DataFrame(per_rows).to_csv(out / "per_detection_predictions.csv", index=False)
    refined_path = out / "refined_plane_frozen_clamp60.txt"
    write_prediction_file(dets, refined_path)
    shutil.copy2(s2_path, out / "baseline_s2_plane.txt")

    # Exact validated evaluator, single frozen refined method only.
    cmd, eval_text = run_validated_evaluator(
        evaluator, gt_dir, s2_path, refined_path, args.score_thr
    )
    (out / "validated_evaluator_output.txt").write_text(eval_text, encoding="utf-8")
    refined_metrics = parse_second_section(eval_text)

    # Geometry diagnostics on fixed baseline TPs.
    pairs = fixed_baseline_matches(dets, gt, 0.50)
    diag_rows = []
    for di,img,gi,biou in pairs:
        d = dets[di]
        g = gt[img][gi]
        riou = polygon_iou(d["refined_poly"], g["poly"])
        before_a = axis_error(d["geom"]["angle_deg"], g["geom"]["angle_deg"])
        after_a = axis_error(d["refined_angle_deg"], g["geom"]["angle_deg"])
        diag_rows.append({
            "image_id": img,
            "det_index": di,
            "gt_index": gi,
            "score": d["score"],
            "gt_aspect_ratio": g["geom"]["aspect_ratio"],
            "s2_aspect_ratio": d["geom"]["aspect_ratio"],
            "raw_delta_deg": d["raw_delta"],
            "used_delta_deg": d["used_delta"],
            "was_clipped": int(abs(d["raw_delta"]) > FROZEN_CLAMP_DEG),
            "angle_error_before_deg": before_a,
            "angle_error_after_deg": after_a,
            "angle_error_gain_deg": before_a-after_a,
            "baseline_iou": biou,
            "refined_iou": riou,
            "iou_gain": riou-biou,
            "baseline_pass75": int(biou >= 0.75),
            "refined_pass75": int(riou >= 0.75),
        })

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out / "matched_diagnostics_baseline_tp.csv", index=False)

    angular = {
        "before_deg": summary_stats(diag["angle_error_before_deg"]),
        "after_deg": summary_stats(diag["angle_error_after_deg"]),
        "gain_deg": summary_stats(diag["angle_error_gain_deg"]),
        "raw_predicted_delta_abs_deg": summary_stats(np.abs(pred_raw)),
        "used_predicted_delta_abs_deg": summary_stats(np.abs(pred_used)),
    }
    iou = {
        "before": summary_stats(diag["baseline_iou"]),
        "after": summary_stats(diag["refined_iou"]),
        "gain": summary_stats(diag["iou_gain"]),
        "pass75_before_percent": float(100*diag["baseline_pass75"].mean()),
        "pass75_after_percent": float(100*diag["refined_pass75"].mean()),
        "pass75_gain_points": float(100*(diag["refined_pass75"].mean()-diag["baseline_pass75"].mean())),
        "improved_gt_0p01_percent": float(100*(diag["iou_gain"] > 0.01).mean()),
        "degraded_lt_minus0p01_percent": float(100*(diag["iou_gain"] < -0.01).mean()),
    }
    jdump(angular, out / "angular_error_summary.json")
    jdump(iou, out / "iou_summary.json")

    # Aspect-ratio diagnostics.
    bins = [
        ("[1.00,1.10)",1.0,1.1),
        ("[1.10,1.20)",1.1,1.2),
        ("[1.20,1.30)",1.2,1.3),
        ("[1.30,1.50)",1.3,1.5),
        ("[1.50,2.00)",1.5,2.0),
        (">=2.00",2.0,float("inf")),
    ]
    ar = []
    for label,lo,hi in bins:
        q = diag[(diag["gt_aspect_ratio"] >= lo) & (diag["gt_aspect_ratio"] < hi)]
        if len(q) == 0:
            continue
        ar.append({
            "aspect_bin": label,
            "n": len(q),
            "median_angle_before_deg": q["angle_error_before_deg"].median(),
            "median_angle_after_deg": q["angle_error_after_deg"].median(),
            "median_iou_before": q["baseline_iou"].median(),
            "median_iou_after": q["refined_iou"].median(),
            "pass75_before_percent": 100*q["baseline_pass75"].mean(),
            "pass75_after_percent": 100*q["refined_pass75"].mean(),
            "pass75_gain_points": 100*(q["refined_pass75"].mean()-q["baseline_pass75"].mean()),
            "improved_iou_gt_0p01_percent": 100*(q["iou_gain"] > 0.01).mean(),
            "degraded_iou_lt_minus0p01_percent": 100*(q["iou_gain"] < -0.01).mean(),
        })
    pd.DataFrame(ar).to_csv(out / "aspect_ratio_diagnostics.csv", index=False)

    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:
        tv = None

    timing = {
        "num_detections": len(dets),
        "wall_seconds_crop_plus_refiner": wall,
        "mean_wall_ms_per_detection_including_crop_io": 1000*wall/max(len(dets),1),
        "peak_cuda_allocated_mib": float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else None,
        "peak_cuda_reserved_mib": float(torch.cuda.max_memory_reserved(device)/1024**2) if device.type=="cuda" else None,
        "note": "Timing includes current Python image loading/crop preprocessing and is not pure ResNet forward latency.",
    }
    jdump(timing, out / "timing.json")

    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": tv,
        "cuda_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    }
    jdump(environment, out / "environment.json")

    summary = {
        "split": "TEST",
        "method_frozen_before_test": True,
        "test_variant_count": 1,
        "frozen_clamp_deg": FROZEN_CLAMP_DEG,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_dev_median_error_deg": float(ckpt.get("best_dev_median_abs_error_deg")),
        "model_total_parameters": total_params,
        "model_trainable_parameters": trainable_params,
        "num_predictions": len(dets),
        "num_clipped_predictions": int(np.sum(np.abs(pred_raw) > FROZEN_CLAMP_DEG)),
        "refined_metrics_validated_evaluator": refined_metrics,
        "angular_diagnostics": angular,
        "iou_diagnostics": iou,
        "timing": timing,
        "evaluator_command": cmd,
    }
    jdump(summary, out / "run_summary.json")

    shutil.copy2(train_script, out / "training_script_snapshot.py")
    shutil.copy2(ckpt_path, out / "best_angle_refiner_snapshot.pt")

    print("\n" + "=" * 92)
    print("FINAL FROZEN TEST RESULT")
    print("=" * 92)
    print(f"AP50      : {refined_metrics['AP50']:.4f}")
    print(f"AP75      : {refined_metrics['AP75']:.4f}")
    print(f"mAP50-95  : {refined_metrics['mAP50_95']:.4f}")
    print(f"Precision : {refined_metrics['precision']:.4f}")
    print(f"Recall    : {refined_metrics['recall']:.4f}")
    print(f"F1        : {refined_metrics['f1']:.4f}")
    print(f"TP/FP/FN  : {refined_metrics['tp']}/{refined_metrics['fp']}/{refined_metrics['fn']}")
    print(f"Angle med : {angular['before_deg']['median']:.4f} -> {angular['after_deg']['median']:.4f} deg")
    print(f"IoU med   : {iou['before']['median']:.4f} -> {iou['after']['median']:.4f}")
    print(f"Pass@.75  : {iou['pass75_before_percent']:.2f}% -> {iou['pass75_after_percent']:.2f}% "
          f"({iou['pass75_gain_points']:+.2f} pts)")
    print(f"Clipped   : {summary['num_clipped_predictions']}/{len(dets)} predictions")
    print(f"Params    : {total_params:,}")
    print(f"Output    : {out}")
    print("Additional TEST variants evaluated : 0")
    print("=" * 92)


if __name__ == "__main__":
    main()
