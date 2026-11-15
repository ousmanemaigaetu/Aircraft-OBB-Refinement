#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 OBB refinement - VAL baseline v0

But:
- partir des prédictions S²A-Net déjà fusionnées sur les images originales;
- calculer l'embedding SAM3 une seule fois par image;
- utiliser chaque OBB S²A-Net comme source d'un prompt XYXY;
- segmenter l'objet avec SAM3;
- convertir le masque en nouvelle OBB via cv2.minAreaRect;
- conserver EXACTEMENT le score S²A-Net afin d'isoler l'effet géométrique;
- ne jamais utiliser les GT pour décider si une boîte est remplacée.

Cette version est volontairement une baseline simple.
Elle travaille uniquement sur VAL et bloque TEST.

Sorties:
- plane_sam3_refined.txt
- diagnostics.csv
- run_summary.txt
"""

from __future__ import annotations

import argparse
import csv
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--sam3-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pred-file", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt-margin", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=0,
                    help="0 = toutes les images. Debug uniquement sinon.")
    return ap.parse_args()


def guard_val_only(*paths: Path):
    for p in paths:
        parts = [x.lower() for x in Path(p).parts]
        if any(x == "test" or x.startswith("test_") or x.endswith("_test") for x in parts):
            raise SystemExit(f"[STOP] Chemin TEST détecté: {p}")


def norm_id(x: str) -> str:
    return Path(str(x)).stem


def find_image(images_dir: Path, image_id: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def read_predictions(path: Path):
    by_image = defaultdict(list)
    total = 0
    invalid = 0

    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        p = line.strip().split()
        if not p:
            continue
        if len(p) < 10:
            invalid += 1
            continue

        try:
            image_id = norm_id(p[0])
            score = float(p[1])
            poly = np.asarray(list(map(float, p[2:10])), dtype=np.float32).reshape(4, 2)
        except Exception:
            invalid += 1
            continue

        by_image[image_id].append({
            "score": score,
            "poly": poly,
            "line_no": line_no,
        })
        total += 1

    return dict(by_image), total, invalid


def poly_area(poly):
    return abs(float(cv2.contourArea(np.asarray(poly, np.float32))))


def poly_center(poly):
    p = np.asarray(poly, np.float32)
    return np.mean(p, axis=0)


def min_area_rect_features(poly):
    rect = cv2.minAreaRect(np.asarray(poly, np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    center = np.asarray(rect[0], np.float32)

    edges = []
    for i in range(4):
        v = box[(i + 1) % 4] - box[i]
        L = float(np.linalg.norm(v))
        ang = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
        edges.append((L, ang))

    major, angle = max(edges, key=lambda z: z[0])
    minor = min(z[0] for z in edges)

    return {
        "box": box,
        "center": center,
        "major": major,
        "minor": minor,
        "angle": angle,
        "area": max(major * minor, 1e-12),
    }


def angle_diff_deg(a, b):
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


def make_prompt_xyxy(poly, width, height, margin):
    x1 = float(np.min(poly[:, 0]))
    y1 = float(np.min(poly[:, 1]))
    x2 = float(np.max(poly[:, 0]))
    y2 = float(np.max(poly[:, 1]))

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    x1 = max(0.0, x1 - margin * bw)
    y1 = max(0.0, y1 - margin * bh)
    x2 = min(float(width - 1), x2 + margin * bw)
    y2 = min(float(height - 1), y2 + margin * bh)

    if x2 <= x1 or y2 <= y1:
        return None

    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def choose_component(mask_bool, reference_center):
    """
    Choix déterministe sans GT:
    1) si le centre de l'OBB S2 tombe dans une composante, on garde celle-ci;
    2) sinon on garde la composante dont le centroïde est le plus proche;
    3) fallback plus grande composante.
    """
    mask8 = (mask_bool.astype(np.uint8) * 255)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask8, connectivity=8)
    if n <= 1:
        return None, "empty_mask"

    cx = int(round(float(reference_center[0])))
    cy = int(round(float(reference_center[1])))

    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        lab = int(labels[cy, cx])
        if lab > 0:
            return (labels == lab).astype(np.uint8) * 255, "center_component"

    valid = np.arange(1, n)
    if len(valid):
        ref = np.asarray(reference_center, dtype=np.float32)
        c = centroids[valid].astype(np.float32)
        dist = np.linalg.norm(c - ref[None, :], axis=1)
        lab = int(valid[int(np.argmin(dist))])
        return (labels == lab).astype(np.uint8) * 255, "nearest_component"

    lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == lab).astype(np.uint8) * 255, "largest_component"


def mask_to_obb(mask8):
    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def format_pred_line(image_id, score, poly):
    vals = " ".join(f"{float(v):.6f}" for v in np.asarray(poly).reshape(-1))
    return f"{image_id} {float(score):.8f} {vals}"


def main():
    args = parse_args()

    guard_val_only(args.pred_file, args.images_dir, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.sam3_repo))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    preds_by_image, n_pred, n_invalid = read_predictions(args.pred_file)
    image_ids = sorted(preds_by_image)

    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]

    print("=" * 78)
    print("SAM3 OBB REFINEMENT - VAL BASELINE V0")
    print("=" * 78)
    print("Images avec prédictions :", len(image_ids))
    print("Prédictions totales     :", n_pred)
    print("Lignes invalides        :", n_invalid)
    print("Prompt margin           :", args.prompt_margin)
    print("Checkpoint              :", args.checkpoint)
    print()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible.")

    print("Chargement SAM3 complet...")
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

    print(f"SAM3 chargé. Temps: {time.time() - t0:.2f} s")
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
    print()

    output_lines = []
    diagnostics = []

    total_done = 0
    total_refined = 0
    total_fallback = 0
    total_errors = 0

    image_times = []

    for image_index, image_id in enumerate(image_ids, 1):
        image_path = find_image(args.images_dir, image_id)
        if image_path is None:
            print(f"[{image_index}/{len(image_ids)}] {image_id}: image introuvable, fallback")
            for j, pred in enumerate(preds_by_image[image_id]):
                output_lines.append(format_pred_line(image_id, pred["score"], pred["poly"]))
                diagnostics.append({
                    "image_id": image_id,
                    "pred_idx": j,
                    "det_score": pred["score"],
                    "sam_score": "",
                    "status": "image_missing",
                    "component_rule": "",
                    "old_area": poly_area(pred["poly"]),
                    "new_area": poly_area(pred["poly"]),
                    "area_ratio": 1.0,
                    "center_shift_norm": 0.0,
                    "angle_shift_deg": 0.0,
                })
                total_fallback += 1
            continue

        t_img = time.time()

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Impossible de lire {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]

        # Une seule extraction de features pour toute l'image.
        inference_state = processor.set_image(Image.fromarray(rgb))

        refined_here = 0

        for j, pred in enumerate(preds_by_image[image_id]):
            old_poly = pred["poly"]
            oldf = min_area_rect_features(old_poly)
            prompt = make_prompt_xyxy(old_poly, w, h, args.prompt_margin)

            status = "ok"
            component_rule = ""
            sam_score = ""
            new_poly = old_poly.copy()

            try:
                if prompt is None:
                    raise ValueError("degenerate_prompt")

                masks, scores, _ = model.predict_inst(
                    inference_state,
                    box=prompt,
                    multimask_output=False,
                    return_logits=False,
                    normalize_coords=True,
                )

                if len(masks) == 0:
                    raise ValueError("no_mask_returned")

                if len(scores):
                    sam_score = float(scores[0])

                mask_bool = np.asarray(masks[0]) > 0

                clean_mask, component_rule = choose_component(
                    mask_bool,
                    oldf["center"],
                )

                if clean_mask is None:
                    raise ValueError("empty_mask")

                candidate = mask_to_obb(clean_mask)
                if candidate is None:
                    raise ValueError("no_valid_contour")

                new_poly = candidate
                total_refined += 1
                refined_here += 1

            except Exception as e:
                status = f"fallback:{type(e).__name__}:{str(e)[:80]}"
                new_poly = old_poly.copy()
                total_fallback += 1
                total_errors += 1

            newf = min_area_rect_features(new_poly)

            diag = math.hypot(oldf["major"], oldf["minor"])
            center_shift = float(np.linalg.norm(newf["center"] - oldf["center"]) / max(diag, 1e-12))
            area_ratio = float(newf["area"] / max(oldf["area"], 1e-12))
            angle_shift = float(angle_diff_deg(newf["angle"], oldf["angle"]))

            output_lines.append(
                format_pred_line(image_id, pred["score"], new_poly)
            )

            diagnostics.append({
                "image_id": image_id,
                "pred_idx": j,
                "det_score": pred["score"],
                "sam_score": sam_score,
                "status": status,
                "component_rule": component_rule,
                "old_area": oldf["area"],
                "new_area": newf["area"],
                "area_ratio": area_ratio,
                "center_shift_norm": center_shift,
                "angle_shift_deg": angle_shift,
            })

            total_done += 1

        dt = time.time() - t_img
        image_times.append(dt)

        print(
            f"[{image_index:02d}/{len(image_ids):02d}] "
            f"{image_id}: {len(preds_by_image[image_id])} preds, "
            f"{refined_here} raffinées, {dt:.2f} s"
        )

    pred_out = args.out_dir / "plane_sam3_refined.txt"
    pred_out.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    csv_out = args.out_dir / "diagnostics.csv"
    if diagnostics:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(diagnostics)

    summary = {
        "images_processed": len(image_ids),
        "predictions_input": n_pred,
        "predictions_processed": total_done,
        "refined": total_refined,
        "fallback": total_fallback,
        "errors": total_errors,
        "prompt_margin": args.prompt_margin,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mean_time_per_image_s": float(np.mean(image_times)) if image_times else None,
    }

    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    txt = [
        "SAM3 OBB REFINEMENT - VAL BASELINE V0",
        "=" * 72,
        f"Images traitées         : {len(image_ids)}",
        f"Prédictions en entrée   : {n_pred}",
        f"Prédictions traitées    : {total_done}",
        f"Raffinements produits   : {total_refined}",
        f"Fallbacks               : {total_fallback}",
        f"Erreurs                 : {total_errors}",
        f"Prompt margin           : {args.prompt_margin}",
        f"Sortie prédictions      : {pred_out}",
        f"Diagnostics             : {csv_out}",
        "",
        "IMPORTANT:",
        "- les scores S²A-Net sont conservés à l'identique;",
        "- aucune GT n'est utilisée pour décider du raffinement;",
        "- cette version est une baseline VAL, pas la configuration finale;",
        "- TEST est bloqué.",
    ]

    summary_txt = "\n".join(txt)
    (args.out_dir / "run_summary.txt").write_text(summary_txt, encoding="utf-8")

    print()
    print(summary_txt)


if __name__ == "__main__":
    main()
