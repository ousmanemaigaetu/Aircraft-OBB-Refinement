#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 2B.1
Measured runtime characterization for YOLO26l-OBB and the ResNet-18 angle refiner.

Run in conda env: yolo26
Hardware target: the same RTX 4090 used for the experiments.

YOLO:
- real TEST patches, batch=1
- deterministic ordered file list
- warm-up before measurement
- Ultralytics preprocess / inference / postprocess timings
- wall-clock per-image latency
- CUDA peak memory

Angle refiner:
- real frozen S2A-Net TEST detections from plane.txt
- original TEST images loaded once
- exact oriented crop geometry from each S2A polygon
- crop extraction and ImageNet preprocessing measured separately
- GPU forward latency batch=1
- GPU throughput batch=64
- postprocessing angle decode + clamp +/-60 measured
- CUDA peak memory
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo-weights", type=Path, required=True)
    p.add_argument("--yolo-test-patches", type=Path, required=True)
    p.add_argument("--refiner-checkpoint", type=Path, required=True)
    p.add_argument("--refiner-script", type=Path, required=True)
    p.add_argument("--s2-test-predictions", type=Path, required=True)
    p.add_argument("--original-test-images", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--yolo-warmup", type=int, default=20)
    p.add_argument("--refiner-warmup", type=int, default=30)
    p.add_argument("--refiner-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stats_ms(xs):
    a = np.asarray(xs, dtype=np.float64)
    if len(a) == 0:
        return {}
    return {
        "n": int(len(a)),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.quantile(a, 0.95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def bytes_to_mib(x):
    return float(x) / (1024.0 ** 2)


def parse_plane_txt(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 10:
                continue
            image_id = sp[0]
            score = float(sp[1])
            poly = np.asarray(list(map(float, sp[2:10])), dtype=np.float32).reshape(4, 2)
            rows.append((image_id, score, poly))
    return rows


def canonical_rect_from_poly(poly):
    (cx, cy), (w, h), angle = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    e01 = box[1] - box[0]
    e12 = box[2] - box[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))
    if l01 >= l12:
        vec = e01
        major, minor = l01, l12
    else:
        vec = e12
        major, minor = l12, l01
    a = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
    return float(cx), float(cy), float(major), float(minor), float(a)


def oriented_square_crop(image_rgb, cx, cy, major, angle_deg,
                         crop_scale=2.0, min_crop_side=64.0, out_size=224):
    side = max(float(min_crop_side), float(crop_scale) * float(major))
    theta = math.radians(float(angle_deg))
    u = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
    v = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    c = np.asarray([cx, cy], dtype=np.float32)
    half = 0.5 * side

    # source: TL, TR, BR in the oriented square
    src = np.stack([
        c - half*u - half*v,
        c + half*u - half*v,
        c + half*u + half*v,
    ]).astype(np.float32)
    dst = np.asarray([
        [0.0, 0.0],
        [out_size - 1.0, 0.0],
        [out_size - 1.0, out_size - 1.0],
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(
        image_rgb,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop


def load_refiner_model(script_path: Path, ckpt_path: Path, device):
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location("refiner_module", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sig = inspect.signature(mod.build_model)
    kwargs = {}
    for name, prm in sig.parameters.items():
        if name.lower() == "pretrained":
            kwargs[name] = False
        elif prm.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(f"Unknown required build_model arg: {name}")

    obj = mod.build_model(**kwargs)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, (list, tuple)):
        model = next(x for x in obj if isinstance(x, torch.nn.Module))
    else:
        raise RuntimeError(f"Unexpected build_model return: {type(obj)}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


def cuda_elapsed_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    for label, p in [
        ("YOLO weights", args.yolo_weights),
        ("YOLO test patches", args.yolo_test_patches),
        ("Refiner checkpoint", args.refiner_checkpoint),
        ("Refiner script", args.refiner_script),
        ("S2 TEST predictions", args.s2_test_predictions),
        ("Original TEST images", args.original_test_images),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    result = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_name,
        }
    }

    # ================================================================
    # YOLO26L-OBB
    # ================================================================
    from ultralytics import YOLO

    patch_paths = sorted(
        p for p in args.yolo_test_patches.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if not patch_paths:
        raise RuntimeError("No YOLO test patch images found.")

    yolo = YOLO(str(args.yolo_weights))

    # Reuse predictor after first construction.
    for i in range(args.yolo_warmup):
        p = patch_paths[i % len(patch_paths)]
        _ = yolo.predict(
            source=str(p),
            imgsz=1024,
            device=0,
            batch=1,
            verbose=False,
            conf=0.25,
            iou=0.7,
            max_det=300,
        )
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    yolo_wall = []
    yolo_pre = []
    yolo_inf = []
    yolo_post = []
    yolo_dets = []

    for p in patch_paths:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rr = yolo.predict(
            source=str(p),
            imgsz=1024,
            device=0,
            batch=1,
            verbose=False,
            conf=0.25,
            iou=0.7,
            max_det=300,
        )
        torch.cuda.synchronize()
        yolo_wall.append((time.perf_counter() - t0) * 1000.0)

        r = rr[0]
        sp = r.speed or {}
        yolo_pre.append(float(sp.get("preprocess", np.nan)))
        yolo_inf.append(float(sp.get("inference", np.nan)))
        yolo_post.append(float(sp.get("postprocess", np.nan)))
        if getattr(r, "obb", None) is not None:
            yolo_dets.append(int(len(r.obb)))
        else:
            yolo_dets.append(0)

    yolo_peak_alloc = torch.cuda.max_memory_allocated()
    yolo_peak_reserved = torch.cuda.max_memory_reserved()

    result["yolo26l_obb"] = {
        "n_test_patches": len(patch_paths),
        "benchmark_settings": {
            "imgsz": 1024,
            "batch": 1,
            "warmup_images": args.yolo_warmup,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 300,
        },
        "wall_clock_per_patch": stats_ms(yolo_wall),
        "ultralytics_preprocess": stats_ms([x for x in yolo_pre if np.isfinite(x)]),
        "ultralytics_inference": stats_ms([x for x in yolo_inf if np.isfinite(x)]),
        "ultralytics_postprocess": stats_ms([x for x in yolo_post if np.isfinite(x)]),
        "throughput_from_mean_wall_patches_per_s": 1000.0 / np.mean(yolo_wall),
        "throughput_from_mean_inference_patches_per_s": 1000.0 / np.mean(yolo_inf),
        "peak_cuda_allocated_mib": bytes_to_mib(yolo_peak_alloc),
        "peak_cuda_reserved_mib": bytes_to_mib(yolo_peak_reserved),
        "detections_per_patch": {
            "mean": float(np.mean(yolo_dets)),
            "median": float(np.median(yolo_dets)),
            "max": int(np.max(yolo_dets)),
        },
    }

    # ================================================================
    # REFINER DATA
    # ================================================================
    detections = parse_plane_txt(args.s2_test_predictions)
    if not detections:
        raise RuntimeError("No S2 TEST detections parsed.")

    image_map = {}
    for p in args.original_test_images.iterdir():
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            image_map[p.stem] = p

    missing = sorted({d[0] for d in detections if d[0] not in image_map})
    if missing:
        raise RuntimeError(f"Missing original images for {len(missing)} ids, first: {missing[:5]}")

    # Load each original image exactly once.
    unique_ids = sorted({d[0] for d in detections})
    decoded = {}
    decode_ms = []
    for image_id in unique_ids:
        t0 = time.perf_counter()
        bgr = cv2.imread(str(image_map[image_id]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imread failed: {image_map[image_id]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decode_ms.append((time.perf_counter() - t0) * 1000.0)
        decoded[image_id] = rgb

    # Exact crop extraction.
    crops = []
    geometry = []
    crop_ms = []
    for image_id, score, poly in detections:
        cx, cy, major, minor, a = canonical_rect_from_poly(poly)
        t0 = time.perf_counter()
        crop = oriented_square_crop(
            decoded[image_id], cx, cy, major, a,
            crop_scale=2.0, min_crop_side=64.0, out_size=224
        )
        crop_ms.append((time.perf_counter() - t0) * 1000.0)
        crops.append(crop)
        geometry.append((image_id, score, cx, cy, major, minor, a))

    # Preprocessing is measured independently.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensors = []
    preprocess_ms = []
    for crop in crops:
        t0 = time.perf_counter()
        x = tfm(Image.fromarray(crop))
        preprocess_ms.append((time.perf_counter() - t0) * 1000.0)
        tensors.append(x)

    model, ref_ckpt = load_refiner_model(
        args.refiner_script, args.refiner_checkpoint, device
    )

    # Warm-up batch=1.
    x0 = tensors[0].unsqueeze(0).to(device, non_blocking=False)
    with torch.no_grad():
        for _ in range(args.refiner_warmup):
            _ = model(x0)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # True GPU forward latency at batch=1, one measurement per real detection.
    ref_b1_ms = []
    raw_outputs = []
    with torch.no_grad():
        for x in tensors:
            xb = x.unsqueeze(0).to(device)
            ms, y = cuda_elapsed_ms(lambda: model(xb))
            ref_b1_ms.append(ms)
            raw_outputs.append(y.detach().cpu())

    ref_b1_peak_alloc = torch.cuda.max_memory_allocated()
    ref_b1_peak_reserved = torch.cuda.max_memory_reserved()

    # Batched throughput, real detection tensors, batch size up to requested batch.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch_times_ms = []
    batch_sizes = []
    batched_outputs = []

    with torch.no_grad():
        for i in range(0, len(tensors), args.refiner_batch):
            xb = torch.stack(tensors[i:i + args.refiner_batch], dim=0).to(device)
            ms, y = cuda_elapsed_ms(lambda: model(xb))
            batch_times_ms.append(ms)
            batch_sizes.append(int(xb.shape[0]))
            batched_outputs.append(y.detach().cpu())

    ref_batch_peak_alloc = torch.cuda.max_memory_allocated()
    ref_batch_peak_reserved = torch.cuda.max_memory_reserved()

    total_batch_s = sum(batch_times_ms) / 1000.0
    batch_throughput = len(tensors) / total_batch_s

    # Decode doubled-angle representation, then clamp +/-60.
    yall = torch.cat(batched_outputs, dim=0).numpy()
    post_ms = []
    used_angles = []
    raw_angles = []
    for y in yall:
        t0 = time.perf_counter()
        # expected output: [cos(2 delta), sin(2 delta)]
        raw = 0.5 * math.degrees(math.atan2(float(y[1]), float(y[0])))
        used = float(np.clip(raw, -60.0, 60.0))
        post_ms.append((time.perf_counter() - t0) * 1000.0)
        raw_angles.append(raw)
        used_angles.append(used)

    n_clipped = int(np.sum(np.abs(np.asarray(raw_angles)) > 60.0))

    # Aggregate optimized refiner overhead over the full TEST set.
    total_decode_ms = float(np.sum(decode_ms))
    total_crop_ms = float(np.sum(crop_ms))
    total_pre_ms = float(np.sum(preprocess_ms))
    total_forward_batched_ms = float(np.sum(batch_times_ms))
    total_post_ms = float(np.sum(post_ms))
    total_refiner_optimized_ms = (
        total_decode_ms + total_crop_ms + total_pre_ms +
        total_forward_batched_ms + total_post_ms
    )

    result["angle_refiner"] = {
        "checkpoint_epoch": ref_ckpt.get("epoch"),
        "n_original_images": len(unique_ids),
        "n_detections": len(detections),
        "detections_per_original_image": float(len(detections) / len(unique_ids)),
        "decode_original_image_once": stats_ms(decode_ms),
        "crop_extraction_per_detection": stats_ms(crop_ms),
        "preprocess_per_detection": stats_ms(preprocess_ms),
        "gpu_forward_batch1_per_detection": stats_ms(ref_b1_ms),
        "gpu_forward_batch1_throughput_detections_per_s": 1000.0 / np.mean(ref_b1_ms),
        "gpu_forward_batch1_peak_allocated_mib": bytes_to_mib(ref_b1_peak_alloc),
        "gpu_forward_batch1_peak_reserved_mib": bytes_to_mib(ref_b1_peak_reserved),
        "gpu_forward_batched": {
            "batch_size_requested": args.refiner_batch,
            "n_batches": len(batch_times_ms),
            "batch_latency_ms": stats_ms(batch_times_ms),
            "effective_throughput_detections_per_s": float(batch_throughput),
            "peak_allocated_mib": bytes_to_mib(ref_batch_peak_alloc),
            "peak_reserved_mib": bytes_to_mib(ref_batch_peak_reserved),
        },
        "postprocess_per_detection": stats_ms(post_ms),
        "n_raw_abs_delta_gt_60": n_clipped,
        "full_test_optimized_pipeline_ms": {
            "image_decode": total_decode_ms,
            "crop_extraction": total_crop_ms,
            "preprocess": total_pre_ms,
            "gpu_forward_batched": total_forward_batched_ms,
            "postprocess": total_post_ms,
            "total": total_refiner_optimized_ms,
            "mean_per_original_image": total_refiner_optimized_ms / len(unique_ids),
            "mean_per_detection": total_refiner_optimized_ms / len(detections),
        },
    }

    # Raw CSVs for later reproducibility.
    import csv

    with (out / "yolo_patch_timings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patch", "wall_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "num_detections"])
        for vals in zip(patch_paths, yolo_wall, yolo_pre, yolo_inf, yolo_post, yolo_dets):
            w.writerow([str(vals[0]), *vals[1:]])

    with (out / "refiner_detection_timings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "crop_ms", "preprocess_ms", "forward_batch1_ms", "postprocess_ms", "raw_delta_deg", "used_delta_deg"])
        for i, g in enumerate(geometry):
            w.writerow([
                g[0], crop_ms[i], preprocess_ms[i], ref_b1_ms[i], post_ms[i],
                raw_angles[i], used_angles[i],
            ])

    full_json = out / "step2b1_full.json"
    full_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    y = result["yolo26l_obb"]
    r = result["angle_refiner"]

    lines = [
        "=" * 96,
        "STEP 2B.1 COMPACT SUMMARY - YOLO26L-OBB + ANGLE REFINER",
        "=" * 96,
        f"GPU: {gpu_name}",
        f"torch: {torch.__version__}",
        "",
        "YOLO26L-OBB",
        f"test_patches: {y['n_test_patches']}",
        f"wall_clock_per_patch: {y['wall_clock_per_patch']}",
        f"preprocess: {y['ultralytics_preprocess']}",
        f"inference: {y['ultralytics_inference']}",
        f"postprocess: {y['ultralytics_postprocess']}",
        f"throughput_wall_patches_s: {y['throughput_from_mean_wall_patches_per_s']}",
        f"throughput_inference_patches_s: {y['throughput_from_mean_inference_patches_per_s']}",
        f"peak_cuda_allocated_mib: {y['peak_cuda_allocated_mib']}",
        f"peak_cuda_reserved_mib: {y['peak_cuda_reserved_mib']}",
        "",
        "ANGLE REFINER",
        f"original_images: {r['n_original_images']}",
        f"detections: {r['n_detections']}",
        f"detections_per_original_image: {r['detections_per_original_image']}",
        f"decode_image_once: {r['decode_original_image_once']}",
        f"crop_per_detection: {r['crop_extraction_per_detection']}",
        f"preprocess_per_detection: {r['preprocess_per_detection']}",
        f"forward_batch1_per_detection: {r['gpu_forward_batch1_per_detection']}",
        f"forward_batch1_throughput_det_s: {r['gpu_forward_batch1_throughput_detections_per_s']}",
        f"forward_batch1_peak_allocated_mib: {r['gpu_forward_batch1_peak_allocated_mib']}",
        f"forward_batched: {r['gpu_forward_batched']}",
        f"postprocess_per_detection: {r['postprocess_per_detection']}",
        f"n_raw_abs_delta_gt_60: {r['n_raw_abs_delta_gt_60']}",
        f"full_test_optimized_pipeline_ms: {r['full_test_optimized_pipeline_ms']}",
        "",
        f"full_json: {full_json}",
        f"yolo_csv: {out / 'yolo_patch_timings.csv'}",
        f"refiner_csv: {out / 'refiner_detection_timings.csv'}",
    ]

    compact = out / "step2b1_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")

    print(compact)
    print("DONE")


if __name__ == "__main__":
    main()
