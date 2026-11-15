#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 OBB refinement - VAL v1 robuste et reprenable

Principes:
- VAL uniquement, TEST bloqué.
- Scores S2A-Net conservés à l'identique.
- Aucune GT utilisée pour accepter/refuser une correction.
- Embedding SAM3 calculé une seule fois par image.
- Nettoyage explicite GPU AVANT et APRES chaque image.
- Sauvegarde par image pour permettre la reprise après interruption/OOM.
- En cas d'OOM persistant sur une image: fallback vers les OBB S2 originales,
  journalisé explicitement, puis poursuite du run.
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--pred-file", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt-margin", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="recalcule même les images déjà terminées")
    return ap.parse_args()


def guard_val_only(*paths):
    for p in paths:
        parts = [x.lower() for x in Path(p).parts]
        if any(x == "test" or x.startswith("test_") or x.endswith("_test") for x in parts):
            raise SystemExit(f"[STOP] Chemin TEST détecté: {p}")


def norm_id(x):
    return Path(str(x)).stem


def find_image(images_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = Path(images_dir) / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def read_predictions(path):
    out = defaultdict(list)
    invalid = 0
    total = 0
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        p = line.strip().split()
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
        out[image_id].append({"score": score, "poly": poly, "line_no": line_no})
        total += 1
    return dict(out), total, invalid


def min_area_rect_features(poly):
    rect = cv2.minAreaRect(np.asarray(poly, np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
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
        "center": np.asarray(rect[0], np.float32),
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
    return np.asarray([x1, y1, x2, y2], np.float32)


def choose_component(mask_bool, reference_center):
    mask8 = mask_bool.astype(np.uint8) * 255
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
    ref = np.asarray(reference_center, np.float32)
    c = centroids[valid].astype(np.float32)
    dist = np.linalg.norm(c - ref[None, :], axis=1)
    lab = int(valid[int(np.argmin(dist))])
    return (labels == lab).astype(np.uint8) * 255, "nearest_component"


def mask_to_obb(mask8):
    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)


def format_pred_line(image_id, score, poly):
    vals = " ".join(f"{float(v):.6f}" for v in np.asarray(poly).reshape(-1))
    return f"{image_id} {float(score):.8f} {vals}"


def cleanup_gpu(*objs):
    # Les objets sont seulement là pour rappeler au lecteur ce qu'on libère.
    # La vraie suppression se fait via del dans la boucle.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def write_csv(path, rows):
    if not rows:
        return
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    args = parse_args()
    guard_val_only(args.pred_file, args.images_dir, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_img_dir = args.out_dir / "per_image"
    per_img_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(args.sam3_repo))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    preds_by_image, n_pred, n_invalid = read_predictions(args.pred_file)
    image_ids = sorted(preds_by_image)
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible")

    print("=" * 78)
    print("SAM3 OBB REFINEMENT - VAL V1 ROBUSTE")
    print("=" * 78)
    print("Images avec prédictions :", len(image_ids))
    print("Prédictions totales     :", n_pred)
    print("Lignes invalides        :", n_invalid)
    print("Prompt margin           :", args.prompt_margin)
    print()

    print("Chargement SAM3...")
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
    print(f"SAM3 chargé en {time.time() - t0:.2f} s")
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
    print()

    run_stats = {
        "images_total": len(image_ids),
        "images_reused": 0,
        "images_computed": 0,
        "images_oom_fallback": 0,
        "predictions_input": n_pred,
        "refined": 0,
        "fallback": 0,
        "errors": 0,
    }

    for idx, image_id in enumerate(image_ids, 1):
        pred_part = per_img_dir / f"{image_id}.txt"
        diag_part = per_img_dir / f"{image_id}.csv"
        meta_part = per_img_dir / f"{image_id}.json"

        if not args.force and pred_part.exists() and diag_part.exists() and meta_part.exists():
            run_stats["images_reused"] += 1
            print(f"[{idx:02d}/{len(image_ids):02d}] {image_id}: déjà terminé, reprise")
            continue

        image_path = find_image(args.images_dir, image_id)
        image_preds = preds_by_image[image_id]

        if image_path is None:
            lines = [format_pred_line(image_id, p["score"], p["poly"]) for p in image_preds]
            diags = []
            for j, p in enumerate(image_preds):
                f = min_area_rect_features(p["poly"])
                diags.append({
                    "image_id": image_id, "pred_idx": j, "det_score": p["score"],
                    "sam_score": "", "status": "image_missing",
                    "component_rule": "", "area_ratio": 1.0,
                    "center_shift_norm": 0.0, "angle_shift_deg": 0.0,
                })
            pred_part.write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_csv(diag_part, diags)
            meta_part.write_text(json.dumps({"status":"image_missing"}, indent=2), encoding="utf-8")
            run_stats["fallback"] += len(image_preds)
            continue

        # CRITIQUE: libérer l'état de l'image précédente AVANT de calculer le suivant.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Impossible de lire {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]

        t_img = time.time()

        inference_state = None
        oom_image = False

        # Essai 1, puis un retry après nettoyage complet.
        for attempt in (1, 2):
            try:
                inference_state = processor.set_image(Image.fromarray(rgb))
                break
            except torch.OutOfMemoryError:
                print(f"  [OOM] {image_id}, tentative {attempt}/2")
                if inference_state is not None:
                    del inference_state
                    inference_state = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                if attempt == 2:
                    oom_image = True

        lines = []
        diags = []
        refined_here = 0
        fallback_here = 0

        if oom_image:
            # Fallback propre: on garde toutes les OBB originales de cette image.
            for j, p in enumerate(image_preds):
                lines.append(format_pred_line(image_id, p["score"], p["poly"]))
                diags.append({
                    "image_id": image_id,
                    "pred_idx": j,
                    "det_score": p["score"],
                    "sam_score": "",
                    "status": "image_oom_fallback",
                    "component_rule": "",
                    "area_ratio": 1.0,
                    "center_shift_norm": 0.0,
                    "angle_shift_deg": 0.0,
                })
            fallback_here = len(image_preds)
            run_stats["images_oom_fallback"] += 1

        else:
            for j, p in enumerate(image_preds):
                old_poly = p["poly"]
                oldf = min_area_rect_features(old_poly)
                prompt = make_prompt_xyxy(old_poly, w, h, args.prompt_margin)

                new_poly = old_poly.copy()
                sam_score = ""
                status = "ok"
                component_rule = ""

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
                        raise ValueError("no_mask")

                    if len(scores):
                        sam_score = float(scores[0])

                    clean, component_rule = choose_component(
                        np.asarray(masks[0]) > 0,
                        oldf["center"],
                    )
                    if clean is None:
                        raise ValueError("empty_mask")

                    candidate = mask_to_obb(clean)
                    if candidate is None:
                        raise ValueError("no_contour")

                    new_poly = candidate
                    refined_here += 1

                except torch.OutOfMemoryError:
                    # Une détection ne doit jamais faire tomber tout le run.
                    status = "fallback:oom_detection"
                    new_poly = old_poly.copy()
                    fallback_here += 1
                    run_stats["errors"] += 1
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    status = f"fallback:{type(e).__name__}:{str(e)[:80]}"
                    new_poly = old_poly.copy()
                    fallback_here += 1
                    run_stats["errors"] += 1

                newf = min_area_rect_features(new_poly)
                diag_len = math.hypot(oldf["major"], oldf["minor"])
                center_shift = float(
                    np.linalg.norm(newf["center"] - oldf["center"]) / max(diag_len, 1e-12)
                )
                area_ratio = float(newf["area"] / max(oldf["area"], 1e-12))
                angle_shift = float(angle_diff_deg(newf["angle"], oldf["angle"]))

                lines.append(format_pred_line(image_id, p["score"], new_poly))
                diags.append({
                    "image_id": image_id,
                    "pred_idx": j,
                    "det_score": p["score"],
                    "sam_score": sam_score,
                    "status": status,
                    "component_rule": component_rule,
                    "area_ratio": area_ratio,
                    "center_shift_norm": center_shift,
                    "angle_shift_deg": angle_shift,
                })

        # Sauvegarde atomique de l'image AVANT de continuer.
        tmp_pred = Path(str(pred_part) + ".tmp")
        tmp_pred.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_pred.replace(pred_part)
        write_csv(diag_part, diags)

        peak_gb = None
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        meta = {
            "image_id": image_id,
            "image_path": str(image_path),
            "num_predictions": len(image_preds),
            "refined": refined_here,
            "fallback": fallback_here,
            "oom_image": oom_image,
            "time_s": time.time() - t_img,
            "peak_cuda_allocated_gb": peak_gb,
        }
        meta_part.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        run_stats["images_computed"] += 1
        run_stats["refined"] += refined_here
        run_stats["fallback"] += fallback_here

        print(
            f"[{idx:02d}/{len(image_ids):02d}] {image_id}: "
            f"{len(image_preds)} preds, {refined_here} raffinées, "
            f"{fallback_here} fallback, {meta['time_s']:.2f}s, "
            f"peak={peak_gb:.2f} GB" if peak_gb is not None else ""
        )

        # CRITIQUE: supprimer l'état GPU avant l'image suivante.
        if inference_state is not None:
            del inference_state
        del bgr, rgb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Assemblage déterministe à la fin.
    final_lines = []
    all_diags = []
    missing_parts = []

    for image_id in image_ids:
        pp = per_img_dir / f"{image_id}.txt"
        dp = per_img_dir / f"{image_id}.csv"
        if not pp.exists() or not dp.exists():
            missing_parts.append(image_id)
            continue

        final_lines.extend(
            [x for x in pp.read_text(encoding="utf-8").splitlines() if x.strip()]
        )
        with open(dp, "r", encoding="utf-8") as f:
            all_diags.extend(list(csv.DictReader(f)))

    if missing_parts:
        raise RuntimeError(f"Images sans sortie finale: {missing_parts}")

    final_pred = args.out_dir / "plane_sam3_refined.txt"
    final_pred.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    final_diag = args.out_dir / "diagnostics.csv"
    write_csv(final_diag, all_diags)

    summary = {
        **run_stats,
        "prompt_margin": args.prompt_margin,
        "checkpoint": str(args.checkpoint),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_prediction_lines": len(final_lines),
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    txt = [
        "SAM3 OBB REFINEMENT - VAL V1 ROBUSTE",
        "=" * 72,
        f"Images totales          : {len(image_ids)}",
        f"Images reprises         : {run_stats['images_reused']}",
        f"Images calculées        : {run_stats['images_computed']}",
        f"Images OOM fallback     : {run_stats['images_oom_fallback']}",
        f"Prédictions entrée      : {n_pred}",
        f"Lignes sortie           : {len(final_lines)}",
        f"Raffinements nouveaux   : {run_stats['refined']}",
        f"Fallbacks nouveaux      : {run_stats['fallback']}",
        f"Erreurs nouvelles       : {run_stats['errors']}",
        f"Prompt margin           : {args.prompt_margin}",
        f"Prédictions finales     : {final_pred}",
        f"Diagnostics             : {final_diag}",
        "",
        "Règles:",
        "- scores S2A-Net inchangés",
        "- aucune GT utilisée dans le raffinement",
        "- sauvegarde par image et reprise automatique",
        "- TEST bloqué",
    ]
    summary_txt = "\n".join(txt)
    (args.out_dir / "run_summary.txt").write_text(summary_txt, encoding="utf-8")
    print()
    print(summary_txt)


if __name__ == "__main__":
    main()
