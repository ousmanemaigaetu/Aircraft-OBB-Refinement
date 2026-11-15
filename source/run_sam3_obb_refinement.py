#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 local-crop OBB refinement for S2A-Net - validation research pipeline.

Key rules:
- validation only; TEST paths are blocked;
- no ground truth is used to decide a refinement;
- S2A-Net confidence scores are preserved exactly;
- each detection is cropped locally before SAM3 inference;
- one SAM3 mask is converted to several controlled geometric hybrids:
    full SAM box
    SAM center only
    SAM angle only
    SAM dimensions only
    SAM center + angle
    SAM center + dimensions
    SAM angle + dimensions
- outputs are saved per image for resume/reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


CONFIGS = {
    "sam_crop_full": (True, True, True),
    "sam_crop_center_only": (True, False, False),
    "sam_crop_angle_only": (False, True, False),
    "sam_crop_dimensions_only": (False, False, True),
    "sam_crop_center_angle": (True, True, False),
    "sam_crop_center_dimensions": (True, False, True),
    "sam_crop_angle_dimensions": (False, True, True),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pred-file", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--crop-scale", type=float, default=3.0,
                    help="square crop side = crop_scale * max(AABB width,height)")
    ap.add_argument("--min-crop-side", type=int, default=96)
    ap.add_argument("--prompt-margin", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-images", type=int, default=0,
                    help="debug only, 0 = all")
    return ap.parse_args()


def guard_val_only(*paths):
    for p in paths:
        parts = [x.lower() for x in Path(p).parts]
        if any(x == "test" or x.startswith("test_") or x.endswith("_test") for x in parts):
            raise SystemExit(f"[STOP] TEST path detected: {p}")


def norm_id(x):
    return Path(str(x)).stem


def read_predictions(path):
    d = defaultdict(list)
    invalid = 0
    total = 0
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        p = line.split()
        if not p:
            continue
        if len(p) < 10:
            invalid += 1
            continue
        try:
            image_id = norm_id(p[0])
            score = float(p[1])
            poly = np.asarray(list(map(float, p[2:10])), np.float32).reshape(4, 2)
        except Exception:
            invalid += 1
            continue
        d[image_id].append({"score": score, "poly": poly, "line_no": line_no})
        total += 1
    return dict(d), total, invalid


def find_image(images_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = Path(images_dir) / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def rect_features(poly):
    rect = cv2.minAreaRect(np.asarray(poly, np.float32))
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
        "area": max(float(major * minor), 1e-12),
    }


def features_to_poly(f):
    th = math.radians(float(f["angle"]))
    u = np.asarray([math.cos(th), math.sin(th)], np.float32)
    v = np.asarray([-math.sin(th), math.cos(th)], np.float32)
    c = np.asarray([f["cx"], f["cy"]], np.float32)
    du = 0.5 * float(f["major"]) * u
    dv = 0.5 * float(f["minor"]) * v
    return np.stack([
        c - du - dv,
        c + du - dv,
        c + du + dv,
        c - du + dv,
    ]).astype(np.float32)


def hybrid(s2f, samf, center=False, angle=False, dimensions=False):
    h = dict(s2f)
    if center:
        h["cx"] = samf["cx"]
        h["cy"] = samf["cy"]
    if angle:
        h["angle"] = samf["angle"]
    if dimensions:
        h["major"] = samf["major"]
        h["minor"] = samf["minor"]
    return h


def angle_diff_deg(a, b):
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


def square_crop_bounds(poly, image_w, image_h, crop_scale, min_side):
    x1 = float(np.min(poly[:, 0]))
    y1 = float(np.min(poly[:, 1]))
    x2 = float(np.max(poly[:, 0]))
    y2 = float(np.max(poly[:, 1]))

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    side = max(float(min_side), float(crop_scale) * max(bw, bh))
    side = min(side, float(image_w), float(image_h))

    left = cx - side / 2
    top = cy - side / 2

    left = min(max(left, 0.0), max(float(image_w) - side, 0.0))
    top = min(max(top, 0.0), max(float(image_h) - side, 0.0))

    right = left + side
    bottom = top + side

    ix1 = int(math.floor(left))
    iy1 = int(math.floor(top))
    ix2 = int(math.ceil(right))
    iy2 = int(math.ceil(bottom))

    ix1 = max(0, min(ix1, image_w - 1))
    iy1 = max(0, min(iy1, image_h - 1))
    ix2 = max(ix1 + 1, min(ix2, image_w))
    iy2 = max(iy1 + 1, min(iy2, image_h))

    return ix1, iy1, ix2, iy2


def local_prompt(poly_global, x0, y0, crop_w, crop_h, margin):
    p = np.asarray(poly_global, np.float32).copy()
    p[:, 0] -= float(x0)
    p[:, 1] -= float(y0)

    x1 = float(np.min(p[:, 0]))
    y1 = float(np.min(p[:, 1]))
    x2 = float(np.max(p[:, 0]))
    y2 = float(np.max(p[:, 1]))

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    x1 -= margin * bw
    y1 -= margin * bh
    x2 += margin * bw
    y2 += margin * bh

    x1 = max(0.0, min(x1, crop_w - 1.0))
    y1 = max(0.0, min(y1, crop_h - 1.0))
    x2 = max(x1 + 1.0, min(x2, crop_w - 1.0))
    y2 = max(y1 + 1.0, min(y2, crop_h - 1.0))

    return np.asarray([x1, y1, x2, y2], np.float32)


def choose_component(mask_bool, ref_center_local):
    mask8 = mask_bool.astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask8, connectivity=8)
    if n <= 1:
        return None, "empty_mask"

    cx = int(round(float(ref_center_local[0])))
    cy = int(round(float(ref_center_local[1])))

    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        lab = int(labels[cy, cx])
        if lab > 0:
            return (labels == lab).astype(np.uint8) * 255, "center_component"

    valid = np.arange(1, n)
    ref = np.asarray(ref_center_local, np.float32)
    c = centroids[valid].astype(np.float32)
    dist = np.linalg.norm(c - ref[None, :], axis=1)
    lab = int(valid[int(np.argmin(dist))])
    return (labels == lab).astype(np.uint8) * 255, "nearest_component"


def mask_to_local_obb(mask8):
    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)


def format_line(image_id, score, poly):
    coords = " ".join(f"{float(v):.6f}" for v in np.asarray(poly).reshape(-1))
    return f"{image_id} {float(score):.8f} {coords}"


def write_csv_atomic(path, rows):
    if not rows:
        return
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    args = parse_args()
    guard_val_only(args.pred_file, args.images_dir, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_image = args.out_dir / "per_image"
    per_image.mkdir(exist_ok=True)

    sys.path.insert(0, str(args.sam3_repo))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    preds, total_preds, invalid = read_predictions(args.pred_file)
    image_ids = sorted(preds)
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]

    print("=" * 80)
    print("SAM3 LOCAL-CROP OBB REFINEMENT - VAL")
    print("=" * 80)
    print("Images                  :", len(image_ids))
    print("Predictions              :", total_preds)
    print("Invalid lines            :", invalid)
    print("Crop scale               :", args.crop_scale)
    print("Minimum crop side        :", args.min_crop_side)
    print("Prompt margin            :", args.prompt_margin)
    print()

    print("Loading SAM3...")
    t0 = time.time()
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        compile=False,
    )
    processor = Sam3Processor(model)
    print(f"SAM3 loaded in {time.time() - t0:.2f}s")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print()

    stats = {
        "images": len(image_ids),
        "predictions": total_preds,
        "refined": 0,
        "fallback": 0,
        "reused_images": 0,
        "computed_images": 0,
    }

    for ii, image_id in enumerate(image_ids, 1):
        image_dir = per_image / image_id
        done_marker = image_dir / "done.json"

        if done_marker.exists() and not args.force:
            stats["reused_images"] += 1
            print(f"[{ii:02d}/{len(image_ids):02d}] {image_id}: resume")
            continue

        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = find_image(args.images_dir, image_id)

        if image_path is None:
            raise RuntimeError(f"Image not found: {image_id}")

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Cannot read: {image_path}")

        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = bgr.shape[:2]

        out_lines = {name: [] for name in CONFIGS}
        diag_rows = []

        t_img = time.time()
        refined_here = 0
        fallback_here = 0

        for jj, p in enumerate(preds[image_id]):
            s2_poly = p["poly"]
            s2f = rect_features(s2_poly)

            x1, y1, x2, y2 = square_crop_bounds(
                s2_poly, W, H, args.crop_scale, args.min_crop_side
            )

            crop_rgb = rgb_full[y1:y2, x1:x2]
            ch, cw = crop_rgb.shape[:2]

            prompt = local_prompt(
                s2_poly, x1, y1, cw, ch, args.prompt_margin
            )

            samf = None
            sam_score = np.nan
            component_rule = ""
            status = "ok"

            try:
                cuda_cleanup()

                inference_state = processor.set_image(
                    Image.fromarray(crop_rgb)
                )

                masks, scores, _ = model.predict_inst(
                    inference_state,
                    box=prompt,
                    multimask_output=False,
                    return_logits=False,
                    normalize_coords=True,
                )

                if len(masks) == 0:
                    raise ValueError("no_mask")

                if len(scores):
                    sam_score = float(scores[0])

                ref_local = np.asarray(
                    [s2f["cx"] - x1, s2f["cy"] - y1],
                    np.float32
                )

                clean_mask, component_rule = choose_component(
                    np.asarray(masks[0]) > 0,
                    ref_local
                )

                if clean_mask is None:
                    raise ValueError("empty_component")

                sam_local_poly = mask_to_local_obb(clean_mask)
                if sam_local_poly is None:
                    raise ValueError("no_contour")

                sam_global_poly = sam_local_poly.copy()
                sam_global_poly[:, 0] += float(x1)
                sam_global_poly[:, 1] += float(y1)

                samf = rect_features(sam_global_poly)
                refined_here += 1

                del inference_state, masks
                cuda_cleanup()

            except torch.OutOfMemoryError:
                status = "fallback:oom"
                fallback_here += 1
                cuda_cleanup()

            except Exception as e:
                status = f"fallback:{type(e).__name__}:{str(e)[:80]}"
                fallback_here += 1
                cuda_cleanup()

            if samf is None:
                samf = dict(s2f)

            for name, (use_center, use_angle, use_dims) in CONFIGS.items():
                hf = hybrid(
                    s2f, samf,
                    center=use_center,
                    angle=use_angle,
                    dimensions=use_dims
                )
                out_lines[name].append(
                    format_line(image_id, p["score"], features_to_poly(hf))
                )

            diag_rows.append({
                "image_id": image_id,
                "pred_idx": jj,
                "det_score": p["score"],
                "sam_score": "" if not np.isfinite(sam_score) else sam_score,
                "status": status,
                "component_rule": component_rule,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "crop_w": cw,
                "crop_h": ch,
                "crop_scale": args.crop_scale,
                "s2_major": s2f["major"],
                "s2_minor": s2f["minor"],
                "sam_major": samf["major"],
                "sam_minor": samf["minor"],
                "area_ratio": samf["area"] / max(s2f["area"], 1e-12),
                "center_shift_norm": math.hypot(
                    samf["cx"] - s2f["cx"],
                    samf["cy"] - s2f["cy"]
                ) / max(math.hypot(s2f["major"], s2f["minor"]), 1e-12),
                "angle_shift_deg": angle_diff_deg(samf["angle"], s2f["angle"]),
            })

        for name, lines in out_lines.items():
            tmp = image_dir / f"{name}.txt.tmp"
            final = image_dir / f"{name}.txt"
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.replace(final)

        write_csv_atomic(image_dir / "diagnostics.csv", diag_rows)

        done_marker.write_text(
            json.dumps({
                "image_id": image_id,
                "predictions": len(preds[image_id]),
                "refined": refined_here,
                "fallback": fallback_here,
                "time_s": time.time() - t_img,
            }, indent=2),
            encoding="utf-8"
        )

        stats["refined"] += refined_here
        stats["fallback"] += fallback_here
        stats["computed_images"] += 1

        print(
            f"[{ii:02d}/{len(image_ids):02d}] {image_id}: "
            f"{len(preds[image_id])} preds, "
            f"{refined_here} refined, "
            f"{fallback_here} fallback, "
            f"{time.time()-t_img:.2f}s"
        )

        del bgr, rgb_full
        cuda_cleanup()

    # Assemble all image-level outputs deterministically.
    for name in CONFIGS:
        all_lines = []
        for image_id in image_ids:
            p = per_image / image_id / f"{name}.txt"
            if not p.exists():
                raise RuntimeError(f"Missing output: {p}")
            all_lines.extend([x for x in p.read_text().splitlines() if x.strip()])
        (args.out_dir / f"{name}.txt").write_text(
            "\n".join(all_lines) + "\n",
            encoding="utf-8"
        )

    all_diag = []
    for image_id in image_ids:
        p = per_image / image_id / "diagnostics.csv"
        with open(p, "r", encoding="utf-8") as f:
            all_diag.extend(list(csv.DictReader(f)))
    write_csv_atomic(args.out_dir / "diagnostics.csv", all_diag)

    summary = {
        **stats,
        "crop_scale": args.crop_scale,
        "min_crop_side": args.min_crop_side,
        "prompt_margin": args.prompt_margin,
        "checkpoint": str(args.checkpoint),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "configurations": list(CONFIGS.keys()),
    }

    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("FIN")
    print("=" * 80)
    print("Refined :", stats["refined"])
    print("Fallback:", stats["fallback"])
    print("Outputs :", args.out_dir)
    print("Configs :", ", ".join(CONFIGS))


if __name__ == "__main__":
    main()
