#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 2B.2
Measured runtime characterization of final S2A-Net on the 846 TEST patches.

Run in conda env: s2anet

Measures on the real TEST patches:
1) full path-based inference wall time
2) image decode time
3) in-memory inference wall time
4) model forward wall time and CUDA-event time
5) overhead outside model forward
6) throughput and CUDA peak memory

The benchmark uses the real inference path, not forward_dummy.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-patches", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


def stats_ms(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean_ms": float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
    }


def mib(x):
    return float(x) / (1024.0 * 1024.0)


def count_detections(result):
    # S2A-Net one-stage result is normally list[class] of ndarray(N, 6)
    if result is None:
        return 0
    if isinstance(result, np.ndarray):
        return int(len(result))
    if isinstance(result, (list, tuple)):
        total = 0
        for x in result:
            if isinstance(x, np.ndarray):
                total += len(x)
            elif isinstance(x, (list, tuple)):
                total += count_detections(x)
        return int(total)
    return 0


def main():
    args = parse_args()

    for label, p in [
        ("repo", args.repo),
        ("config", args.config),
        ("checkpoint", args.checkpoint),
        ("test patches", args.test_patches),
    ]:
        if not p.exists():
            raise FileNotFoundError("{} not found: {}".format(label, p))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.repo.resolve()))

    import mmcv
    import mmdet
    from mmdet.apis import init_detector, inference_detector

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    gpu = torch.cuda.get_device_name(0)

    print("=" * 96)
    print("STEP 2B.2 - S2A-NET REAL TEST RUNTIME")
    print("=" * 96)
    print("GPU:", gpu)
    print("torch:", torch.__version__)
    print("mmcv:", getattr(mmcv, "__version__", None))
    print("mmdet:", getattr(mmdet, "__version__", None))

    model = init_detector(
        str(args.config),
        str(args.checkpoint),
        device="cuda:0"
    )
    model.eval()

    patch_paths = sorted([
        p for p in args.test_patches.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ])

    if not patch_paths:
        raise RuntimeError("No test patch images found")

    # ------------------------------------------------------------------
    # Instrument the REAL model.forward called by inference_detector.
    # This excludes file decode and preprocessing outside the detector,
    # but includes the detector's test-time forward and internal postprocess.
    # ------------------------------------------------------------------
    original_forward = model.forward
    forward_wall_log = []
    forward_cuda_log = []

    def timed_forward(*a, **kw):
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        out = original_forward(*a, **kw)
        ev1.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        forward_wall_log.append((t1 - t0) * 1000.0)
        forward_cuda_log.append(float(ev0.elapsed_time(ev1)))
        return out

    model.forward = timed_forward

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    warm_img = cv2.imread(str(patch_paths[0]), cv2.IMREAD_COLOR)
    if warm_img is None:
        raise RuntimeError("Cannot decode warmup image")

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = inference_detector(model, warm_img)
    torch.cuda.synchronize()

    forward_wall_log[:] = []
    forward_cuda_log[:] = []

    # ------------------------------------------------------------------
    # PASS A: complete path-based inference, one patch at a time
    # ------------------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    full_path_ms = []
    full_forward_wall = []
    full_forward_cuda = []
    det_counts = []

    with torch.no_grad():
        for p in patch_paths:
            n0 = len(forward_wall_log)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = inference_detector(model, str(p))
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            full_path_ms.append((t1 - t0) * 1000.0)

            if len(forward_wall_log) != n0 + 1:
                raise RuntimeError("Unexpected number of model.forward calls")
            full_forward_wall.append(forward_wall_log[-1])
            full_forward_cuda.append(forward_cuda_log[-1])
            det_counts.append(count_detections(result))

    peak_full_alloc = torch.cuda.max_memory_allocated()
    peak_full_reserved = torch.cuda.max_memory_reserved()

    # ------------------------------------------------------------------
    # PASS B: explicitly separate decode from in-memory inference
    # ------------------------------------------------------------------
    forward_wall_log[:] = []
    forward_cuda_log[:] = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    decode_ms = []
    in_memory_ms = []
    mem_forward_wall = []
    mem_forward_cuda = []
    outside_forward_ms = []

    with torch.no_grad():
        for p in patch_paths:
            t0 = time.perf_counter()
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            t1 = time.perf_counter()
            if img is None:
                raise RuntimeError("cv2.imread failed: {}".format(p))
            decode_ms.append((t1 - t0) * 1000.0)

            n0 = len(forward_wall_log)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            _ = inference_detector(model, img)
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            this_mem = (t3 - t2) * 1000.0
            in_memory_ms.append(this_mem)

            if len(forward_wall_log) != n0 + 1:
                raise RuntimeError("Unexpected number of model.forward calls in memory pass")

            fw = forward_wall_log[-1]
            fc = forward_cuda_log[-1]
            mem_forward_wall.append(fw)
            mem_forward_cuda.append(fc)
            outside_forward_ms.append(max(0.0, this_mem - fw))

    peak_mem_alloc = torch.cuda.max_memory_allocated()
    peak_mem_reserved = torch.cuda.max_memory_reserved()

    result = {
        "environment": {
            "gpu": gpu,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mmcv": getattr(mmcv, "__version__", None),
            "mmdet": getattr(mmdet, "__version__", None),
        },
        "settings": {
            "n_test_patches": len(patch_paths),
            "warmup": args.warmup,
            "batch": 1,
            "real_inference": True,
            "input_patch_size": "1024x1024 dataset patches",
        },
        "full_path_inference_per_patch": stats_ms(full_path_ms),
        "full_path_throughput_patches_s": 1000.0 / float(np.mean(full_path_ms)),
        "full_path_forward_wall_per_patch": stats_ms(full_forward_wall),
        "full_path_forward_cuda_per_patch": stats_ms(full_forward_cuda),
        "full_path_peak_cuda_allocated_mib": mib(peak_full_alloc),
        "full_path_peak_cuda_reserved_mib": mib(peak_full_reserved),
        "decode_per_patch": stats_ms(decode_ms),
        "in_memory_inference_per_patch": stats_ms(in_memory_ms),
        "in_memory_throughput_patches_s": 1000.0 / float(np.mean(in_memory_ms)),
        "model_forward_wall_per_patch": stats_ms(mem_forward_wall),
        "model_forward_cuda_per_patch": stats_ms(mem_forward_cuda),
        "outside_model_forward_per_patch": stats_ms(outside_forward_ms),
        "in_memory_peak_cuda_allocated_mib": mib(peak_mem_alloc),
        "in_memory_peak_cuda_reserved_mib": mib(peak_mem_reserved),
        "patch_detection_count": {
            "mean": float(np.mean(det_counts)),
            "median": float(np.median(det_counts)),
            "max": int(np.max(det_counts)),
            "sum": int(np.sum(det_counts)),
        },
    }

    full_json = args.out_dir / "step2b2_full.json"
    full_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = args.out_dir / "s2anet_patch_timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "patch", "full_path_ms", "full_forward_wall_ms", "full_forward_cuda_ms",
            "decode_ms", "in_memory_ms", "model_forward_wall_ms",
            "model_forward_cuda_ms", "outside_forward_ms", "num_detections"
        ])
        for i, p in enumerate(patch_paths):
            w.writerow([
                str(p),
                full_path_ms[i],
                full_forward_wall[i],
                full_forward_cuda[i],
                decode_ms[i],
                in_memory_ms[i],
                mem_forward_wall[i],
                mem_forward_cuda[i],
                outside_forward_ms[i],
                det_counts[i],
            ])

    lines = [
        "=" * 96,
        "STEP 2B.2 COMPACT SUMMARY - S2A-NET",
        "=" * 96,
        "GPU: {}".format(gpu),
        "torch: {}".format(torch.__version__),
        "mmcv: {}".format(getattr(mmcv, "__version__", None)),
        "mmdet: {}".format(getattr(mmdet, "__version__", None)),
        "",
        "test_patches: {}".format(len(patch_paths)),
        "warmup: {}".format(args.warmup),
        "batch: 1",
        "",
        "FULL PATH",
        "full_path_per_patch: {}".format(result["full_path_inference_per_patch"]),
        "full_path_throughput_patches_s: {}".format(result["full_path_throughput_patches_s"]),
        "forward_wall_inside_full_path: {}".format(result["full_path_forward_wall_per_patch"]),
        "forward_cuda_inside_full_path: {}".format(result["full_path_forward_cuda_per_patch"]),
        "peak_cuda_allocated_mib: {}".format(result["full_path_peak_cuda_allocated_mib"]),
        "peak_cuda_reserved_mib: {}".format(result["full_path_peak_cuda_reserved_mib"]),
        "",
        "DECOMPOSED IN-MEMORY PASS",
        "decode_per_patch: {}".format(result["decode_per_patch"]),
        "in_memory_inference_per_patch: {}".format(result["in_memory_inference_per_patch"]),
        "in_memory_throughput_patches_s: {}".format(result["in_memory_throughput_patches_s"]),
        "model_forward_wall_per_patch: {}".format(result["model_forward_wall_per_patch"]),
        "model_forward_cuda_per_patch: {}".format(result["model_forward_cuda_per_patch"]),
        "outside_model_forward_per_patch: {}".format(result["outside_model_forward_per_patch"]),
        "peak_cuda_allocated_mib: {}".format(result["in_memory_peak_cuda_allocated_mib"]),
        "peak_cuda_reserved_mib: {}".format(result["in_memory_peak_cuda_reserved_mib"]),
        "",
        "PATCH DETECTIONS",
        "{}".format(result["patch_detection_count"]),
        "",
        "full_json: {}".format(full_json),
        "csv: {}".format(csv_path),
    ]

    compact = args.out_dir / "step2b2_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")

    print(compact)
    print("DONE")


if __name__ == "__main__":
    main()
