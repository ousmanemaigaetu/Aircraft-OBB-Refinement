#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 3
Final frozen TEST geometric analysis.

READ-ONLY. Uses ONLY already-produced TEST predictions.
No threshold search, no tuning, no model inference.

Protocol:
- baseline S2A-Net predictions at the already-frozen score threshold 0.50
- one-to-one greedy matching to useful TEST GT at IoU >= 0.50
- difficult GT are not used as geometry targets
- the SAME baseline prediction <-> GT pairing is kept for the refined polygon
- score, image id, and prediction row correspondence must be identical
- reports geometry before/after without rematching the refined predictions

Outputs:
- step3_test_geometry_compact_summary.txt
- step3_test_geometry_full.json
- step3_test_geometry_per_match.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SCORE_THR = 0.50
MATCH_IOU_THR = 0.50
STABLE_IOU_EPS = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gt-dir", type=Path,
        default=Path("dataset_avions/test/annotations")
    )
    p.add_argument(
        "--baseline", type=Path,
        default=Path("modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/test_final/results_after_nms/plane.txt")
    )
    p.add_argument(
        "--refined", type=Path, default=None,
        help="Final frozen refined plane.txt. If omitted, strict auto-discovery is attempted."
    )
    p.add_argument(
        "--per-detection-csv", type=Path, default=None,
        help="Frozen refiner per-detection CSV with raw/used delta. Optional strict auto-discovery."
    )
    p.add_argument(
        "--search-root", type=Path,
        default=Path("runs/angle_refiner/resnet18_angle")
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("runs/final_test_geometry/step3")
    )
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def poly_iou(a, b):
    a = np.asarray(a, np.float32).reshape(4, 2)
    b = np.asarray(b, np.float32).reshape(4, 2)
    aa = abs(float(cv2.contourArea(a)))
    bb = abs(float(cv2.contourArea(b)))
    if aa <= 0 or bb <= 0:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
        inter = max(float(inter), 0.0)
    except cv2.error:
        return 0.0
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def canonical_rect(poly):
    p = np.asarray(poly, np.float32).reshape(4, 2)
    (cx, cy), (w, h), a = cv2.minAreaRect(p)
    box = cv2.boxPoints(((cx, cy), (w, h), a)).astype(np.float64)
    e01 = box[1] - box[0]
    e12 = box[2] - box[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))
    if l01 >= l12:
        major, minor, v = l01, l12, e01
    else:
        major, minor, v = l12, l01, e12
    angle = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
    return {
        "cx": float(cx), "cy": float(cy),
        "major": major, "minor": minor,
        "angle": angle,
        "aspect": major / minor if minor > 0 else float("inf"),
        "area_rect": major * minor,
        "size_sqrt_area": math.sqrt(max(major * minor, 0.0)),
    }


def axis_error_deg(a, b):
    d = abs((float(a) - float(b)) % 180.0)
    return min(d, 180.0 - d)


def parse_predictions(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.strip().split()
            if len(s) < 10:
                continue
            rows.append({
                "row": idx,
                "image_id": s[0],
                "score": float(s[1]),
                "poly": np.asarray([float(x) for x in s[2:10]], dtype=np.float64).reshape(4, 2)
            })
    return rows


def parse_gt_dir(gt_dir):
    data = {}
    for p in sorted(gt_dir.glob("*.txt")):
        useful, difficult = [], []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip().split()
            if len(s) < 9:
                continue
            poly = np.asarray([float(x) for x in s[:8]], dtype=np.float64).reshape(4, 2)
            cls = s[8]
            difficulty = int(s[9]) if len(s) >= 10 else 0
            if cls != "plane":
                continue
            obj = {"poly": poly, "difficulty": difficulty}
            if difficulty == 0:
                useful.append(obj)
            else:
                difficult.append(obj)
        data[p.stem] = {"useful": useful, "difficult": difficult}
    return data


def signature(rows):
    return [(r["image_id"], round(r["score"], 12)) for r in rows]


def strict_autodiscover_refined(search_root, baseline_rows, baseline_path):
    candidates = []
    base_sig = signature(baseline_rows)

    for p in search_root.rglob("plane.txt"):
        try:
            if p.resolve() == baseline_path.resolve():
                continue
            rows = parse_predictions(p)
            if len(rows) != len(baseline_rows):
                continue
            if signature(rows) != base_sig:
                continue
            low = str(p).lower()
            rank = (
                int("frozen" in low) * 8 +
                int("clamp60" in low or "clamp_60" in low or "clamp-60" in low) * 8 +
                int("final" in low) * 4 +
                int("refin" in low) * 2
            )
            candidates.append((rank, p, sha256(p)))
        except Exception:
            pass

    if not candidates:
        raise RuntimeError(
            "No refined plane.txt with the same 1100-row id/score signature as baseline was found."
        )

    best_rank = max(x[0] for x in candidates)
    best = [x for x in candidates if x[0] == best_rank]
    unique_hashes = sorted(set(x[2] for x in best))

    if len(unique_hashes) != 1:
        msg = "\n".join("{} | {} | {}".format(r, h, p) for r, p, h in best)
        raise RuntimeError(
            "Ambiguous final refined prediction candidates with different contents:\n" + msg
        )

    chosen = sorted([x for x in best if x[2] == unique_hashes[0]], key=lambda x: str(x[1]))[0][1]
    return chosen, candidates


def autodiscover_per_detection(search_root):
    cands = []
    for p in search_root.rglob("per_detection_predictions.csv"):
        low = str(p).lower()
        rank = (
            int("frozen" in low) * 8 +
            int("clamp60" in low or "clamp_60" in low or "clamp-60" in low) * 8 +
            int("final" in low) * 4
        )
        cands.append((rank, p))
    if not cands:
        return None
    best_rank = max(x[0] for x in cands)
    best = sorted([p for r, p in cands if r == best_rank], key=str)
    return best[0]


def dist_summary(x):
    a = np.asarray(list(x), dtype=float)
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


def summarize_subset(df):
    if len(df) == 0:
        return {"n": 0}
    gain = df["iou_refined"] - df["iou_baseline"]
    return {
        "n": int(len(df)),
        "angle_baseline_deg": dist_summary(df["angle_error_baseline_deg"]),
        "angle_refined_deg": dist_summary(df["angle_error_refined_deg"]),
        "iou_baseline": dist_summary(df["iou_baseline"]),
        "iou_refined": dist_summary(df["iou_refined"]),
        "iou_gain": dist_summary(gain),
        "pass75_baseline_pct": float(100.0 * np.mean(df["iou_baseline"] >= 0.75)),
        "pass75_refined_pct": float(100.0 * np.mean(df["iou_refined"] >= 0.75)),
    }


def aspect_bin(ar):
    if ar < 1.1: return "[1.0,1.1)"
    if ar < 1.2: return "[1.1,1.2)"
    if ar < 1.3: return "[1.2,1.3)"
    if ar < 1.5: return "[1.3,1.5)"
    if ar < 2.0: return "[1.5,2.0)"
    return ">=2.0"


def size_bin(s):
    if s < 32: return "<32"
    if s < 64: return "[32,64)"
    if s < 128: return "[64,128)"
    return ">=128"


def match_baseline_to_gt(baseline_rows, gt):
    # Keep only frozen score threshold, then greedy score-descending per image.
    by_image = {}
    for r in baseline_rows:
        if r["score"] >= SCORE_THR:
            by_image.setdefault(r["image_id"], []).append(r)

    matches = []
    fp = 0
    ignored = 0

    for image_id, preds in by_image.items():
        preds = sorted(preds, key=lambda x: (-x["score"], x["row"]))
        useful = gt.get(image_id, {}).get("useful", [])
        difficult = gt.get(image_id, {}).get("difficult", [])
        used = set()

        for pred in preds:
            if useful:
                ious = [poly_iou(pred["poly"], g["poly"]) for g in useful]
                order = np.argsort(ious)[::-1]
            else:
                ious, order = [], []

            chosen = None
            for j in order:
                j = int(j)
                if ious[j] < MATCH_IOU_THR:
                    break
                if j not in used:
                    chosen = j
                    break

            if chosen is not None:
                used.add(chosen)
                matches.append((pred, chosen))
                continue

            # Ignore prediction if it overlaps a difficult GT at threshold.
            max_diff = max([poly_iou(pred["poly"], g["poly"]) for g in difficult], default=0.0)
            if max_diff >= MATCH_IOU_THR:
                ignored += 1
            else:
                fp += 1

    total_useful = sum(len(v["useful"]) for v in gt.values())
    tp = len(matches)
    fn = total_useful - tp
    return matches, {"tp": tp, "fp": fp, "fn": fn, "ignored_predictions": ignored}


def clamp_stats(path):
    if path is None or not path.is_file():
        return None

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    raw_candidates = [
        "raw_delta_deg", "pred_delta_deg", "delta_raw_deg",
        "raw_angle_delta_deg", "predicted_delta_deg",
    ]
    used_candidates = [
        "used_delta_deg", "clamped_delta_deg", "delta_used_deg",
        "final_delta_deg",
    ]

    raw_col = next((cols[x] for x in raw_candidates if x in cols), None)
    used_col = next((cols[x] for x in used_candidates if x in cols), None)

    if raw_col is None:
        # fuzzy fallback
        for c in df.columns:
            l = c.lower()
            if "raw" in l and "delta" in l:
                raw_col = c
                break

    if used_col is None:
        for c in df.columns:
            l = c.lower()
            if ("used" in l or "clamp" in l or "final" in l) and "delta" in l:
                used_col = c
                break

    if raw_col is None:
        return {
            "csv": str(path),
            "rows": int(len(df)),
            "raw_col": None,
            "used_col": used_col,
            "error": "Could not identify raw delta column"
        }

    raw = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(float)
    n_valid = int(np.isfinite(raw).sum())
    n_clip = int(np.sum(np.abs(raw[np.isfinite(raw)]) > 60.0))

    out = {
        "csv": str(path),
        "rows": int(len(df)),
        "raw_col": raw_col,
        "used_col": used_col,
        "valid_raw_rows": n_valid,
        "raw_abs_gt_60_count": n_clip,
        "raw_abs_gt_60_pct": float(100.0 * n_clip / n_valid) if n_valid else None,
    }

    if used_col is not None:
        used = pd.to_numeric(df[used_col], errors="coerce").to_numpy(float)
        valid = np.isfinite(raw) & np.isfinite(used)
        expected = np.clip(raw[valid], -60.0, 60.0)
        out["clamp_max_abs_error_deg"] = float(np.max(np.abs(expected - used[valid]))) if valid.any() else None

    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for label, p in [("GT dir", args.gt_dir), ("baseline", args.baseline)]:
        if not p.exists():
            raise FileNotFoundError("{} not found: {}".format(label, p))

    baseline = parse_predictions(args.baseline)
    if len(baseline) != 1100:
        raise RuntimeError("Expected 1100 final baseline rows, got {}".format(len(baseline)))

    if args.refined is None:
        refined_path, discovery = strict_autodiscover_refined(
            args.search_root, baseline, args.baseline
        )
    else:
        refined_path = args.refined
        discovery = []
        if not refined_path.is_file():
            raise FileNotFoundError("refined not found: {}".format(refined_path))

    refined = parse_predictions(refined_path)

    if len(refined) != len(baseline):
        raise RuntimeError("Baseline/refined row count mismatch")

    # Exact row correspondence check.
    id_mismatch = 0
    score_max_diff = 0.0
    for b, r in zip(baseline, refined):
        if b["image_id"] != r["image_id"]:
            id_mismatch += 1
        score_max_diff = max(score_max_diff, abs(b["score"] - r["score"]))

    if id_mismatch != 0 or score_max_diff > 1e-10:
        raise RuntimeError(
            "Prediction correspondence failed: id_mismatch={}, score_max_diff={}".format(
                id_mismatch, score_max_diff
            )
        )

    gt = parse_gt_dir(args.gt_dir)
    matches, eval_counts = match_baseline_to_gt(baseline, gt)

    # Critical guard: must reproduce the frozen TEST operating point exactly.
    expected = {"tp": 691, "fp": 23, "fn": 65}
    for k, v in expected.items():
        if eval_counts[k] != v:
            raise RuntimeError(
                "Baseline TEST matching does not reproduce expected {}={} (got {}). "
                "Stop rather than analyzing a mismatched protocol.".format(k, v, eval_counts[k])
            )

    refined_by_row = {r["row"]: r for r in refined}

    records = []
    for pred, gt_idx in matches:
        image_id = pred["image_id"]
        gpoly = gt[image_id]["useful"][gt_idx]["poly"]
        rpred = refined_by_row[pred["row"]]

        g = canonical_rect(gpoly)
        b = canonical_rect(pred["poly"])
        rr = canonical_rect(rpred["poly"])

        iou_b = poly_iou(pred["poly"], gpoly)
        iou_r = poly_iou(rpred["poly"], gpoly)

        rec = {
            "image_id": image_id,
            "prediction_row": pred["row"],
            "score": pred["score"],
            "gt_index": gt_idx,
            "iou_baseline": iou_b,
            "iou_refined": iou_r,
            "iou_gain": iou_r - iou_b,
            "angle_error_baseline_deg": axis_error_deg(b["angle"], g["angle"]),
            "angle_error_refined_deg": axis_error_deg(rr["angle"], g["angle"]),
            "angle_error_gain_deg": axis_error_deg(b["angle"], g["angle"]) - axis_error_deg(rr["angle"], g["angle"]),
            "gt_major_px": g["major"],
            "gt_minor_px": g["minor"],
            "gt_aspect_ratio": g["aspect"],
            "gt_size_sqrt_area_px": g["size_sqrt_area"],
            "aspect_bin": aspect_bin(g["aspect"]),
            "size_bin": size_bin(g["size_sqrt_area"]),
            "baseline_pass75": bool(iou_b >= 0.75),
            "refined_pass75": bool(iou_r >= 0.75),
        }
        records.append(rec)

    df = pd.DataFrame(records)

    gain = df["iou_gain"].to_numpy()
    improved = int(np.sum(gain > STABLE_IOU_EPS))
    degraded = int(np.sum(gain < -STABLE_IOU_EPS))
    stable = int(len(df) - improved - degraded)

    up75 = int(np.sum((df["iou_baseline"] < 0.75) & (df["iou_refined"] >= 0.75)))
    down75 = int(np.sum((df["iou_baseline"] >= 0.75) & (df["iou_refined"] < 0.75)))

    aspect_order = ["[1.0,1.1)", "[1.1,1.2)", "[1.2,1.3)", "[1.3,1.5)", "[1.5,2.0)", ">=2.0"]
    size_order = ["<32", "[32,64)", "[64,128)", ">=128"]

    by_aspect = {b: summarize_subset(df[df["aspect_bin"] == b]) for b in aspect_order}
    by_size = {b: summarize_subset(df[df["size_bin"] == b]) for b in size_order}

    per_csv = args.per_detection_csv
    if per_csv is None:
        per_csv = autodiscover_per_detection(args.search_root)

    clamp = clamp_stats(per_csv)

    full = {
        "protocol": {
            "dataset": "TEST",
            "score_threshold": SCORE_THR,
            "baseline_match_iou_threshold": MATCH_IOU_THR,
            "pairing": "baseline prediction to useful GT; identical pair retained for refined comparison; no refined rematching",
            "stable_iou_epsilon": STABLE_IOU_EPS,
            "no_test_tuning": True,
        },
        "files": {
            "baseline": str(args.baseline),
            "baseline_sha256": sha256(args.baseline),
            "refined": str(refined_path),
            "refined_sha256": sha256(refined_path),
            "per_detection_csv": str(per_csv) if per_csv else None,
        },
        "integrity": {
            "baseline_rows": len(baseline),
            "refined_rows": len(refined),
            "image_id_mismatch_count": id_mismatch,
            "score_max_abs_diff": score_max_diff,
            "baseline_operating_point_counts": eval_counts,
            "expected_counts": expected,
        },
        "matched_geometry": {
            "n": int(len(df)),
            "angle_baseline_deg": dist_summary(df["angle_error_baseline_deg"]),
            "angle_refined_deg": dist_summary(df["angle_error_refined_deg"]),
            "angle_error_improvement_deg": dist_summary(df["angle_error_gain_deg"]),
            "iou_baseline": dist_summary(df["iou_baseline"]),
            "iou_refined": dist_summary(df["iou_refined"]),
            "iou_gain": dist_summary(df["iou_gain"]),
            "improved_iou_gt_0p01": improved,
            "degraded_iou_lt_minus_0p01": degraded,
            "stable_abs_iou_gain_le_0p01": stable,
            "improved_pct": float(100.0 * improved / len(df)),
            "degraded_pct": float(100.0 * degraded / len(df)),
            "stable_pct": float(100.0 * stable / len(df)),
            "pass75_baseline_count": int(np.sum(df["baseline_pass75"])),
            "pass75_refined_count": int(np.sum(df["refined_pass75"])),
            "pass75_baseline_pct": float(100.0 * np.mean(df["baseline_pass75"])),
            "pass75_refined_pct": float(100.0 * np.mean(df["refined_pass75"])),
            "cross_up_75": up75,
            "cross_down_75": down75,
            "net_cross_75": up75 - down75,
        },
        "by_aspect_ratio": by_aspect,
        "by_size_sqrt_area_px": by_size,
        "clamp": clamp,
        "notes": [
            "These are conditional geometric diagnostics on baseline TEST true positives at the frozen score 0.50 operating point.",
            "Pass75 in this file is NOT AP75.",
            "Raw DOTA polygons are used for IoU; minAreaRect is used only for canonical angle/aspect/size descriptors.",
            "The refined detections are not rematched, preventing pair-selection changes from contaminating before/after geometry.",
        ],
    }

    out_json = args.out_dir / "step3_test_geometry_full.json"
    out_csv = args.out_dir / "step3_test_geometry_per_match.csv"
    out_txt = args.out_dir / "step3_test_geometry_compact_summary.txt"

    out_json.write_text(json.dumps(full, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    m = full["matched_geometry"]
    lines = [
        "=" * 100,
        "STEP 3 FINAL TEST GEOMETRIC ANALYSIS",
        "=" * 100,
        "baseline: {}".format(args.baseline),
        "refined: {}".format(refined_path),
        "baseline_sha256: {}".format(full["files"]["baseline_sha256"]),
        "refined_sha256: {}".format(full["files"]["refined_sha256"]),
        "",
        "INTEGRITY / FROZEN OPERATING POINT",
        "baseline_rows: {}".format(len(baseline)),
        "refined_rows: {}".format(len(refined)),
        "image_id_mismatch_count: {}".format(id_mismatch),
        "score_max_abs_diff: {}".format(score_max_diff),
        "TP/FP/FN: {}/{}/{}".format(eval_counts["tp"], eval_counts["fp"], eval_counts["fn"]),
        "ignored_predictions_on_difficult: {}".format(eval_counts["ignored_predictions"]),
        "",
        "MATCHED BASELINE TRUE POSITIVES - SAME PAIRS BEFORE/AFTER",
        "n: {}".format(m["n"]),
        "angle_baseline_deg: {}".format(m["angle_baseline_deg"]),
        "angle_refined_deg: {}".format(m["angle_refined_deg"]),
        "angle_error_improvement_deg: {}".format(m["angle_error_improvement_deg"]),
        "iou_baseline: {}".format(m["iou_baseline"]),
        "iou_refined: {}".format(m["iou_refined"]),
        "iou_gain: {}".format(m["iou_gain"]),
        "improved/degraded/stable (>0.01 / <-0.01 / otherwise): {}/{}/{}".format(
            improved, degraded, stable
        ),
        "improved/degraded/stable pct: {:.3f}/{:.3f}/{:.3f}".format(
            m["improved_pct"], m["degraded_pct"], m["stable_pct"]
        ),
        "pass75 baseline/refined: {}/{} ({:.3f}% -> {:.3f}%)".format(
            m["pass75_baseline_count"], m["pass75_refined_count"],
            m["pass75_baseline_pct"], m["pass75_refined_pct"]
        ),
        "cross_up_75: {}".format(up75),
        "cross_down_75: {}".format(down75),
        "net_cross_75: {}".format(up75 - down75),
        "",
        "ASPECT RATIO BINS",
    ]

    for b in aspect_order:
        lines.append("{}: {}".format(b, by_aspect[b]))

    lines.append("")
    lines.append("SIZE BINS sqrt(GT rectangle area) [px]")
    for b in size_order:
        lines.append("{}: {}".format(b, by_size[b]))

    lines += [
        "",
        "CLAMP",
        "{}".format(clamp),
        "",
        "IMPORTANT: conditional pass75 is NOT AP75.",
        "No refined rematching and no TEST tuning were performed.",
        "",
        "json: {}".format(out_json),
        "csv: {}".format(out_csv),
    ]

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(out_txt)
    print("DONE")


if __name__ == "__main__":
    main()
