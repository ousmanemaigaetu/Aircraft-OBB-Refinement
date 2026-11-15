#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Scientific integrity audit, Step 1C
Geometry, angular conventions, difficult handling and evaluation protocol.

READ-ONLY with respect to project data/models/results.
Only writes audit outputs and temporary synthetic audit inputs under out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--dataset-root", type=Path, default=Path("dataset_avions"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/scientific_audit/step1c_geometry_protocol"),
    )
    p.add_argument("--evaluator", type=Path, default=Path("eval_obb_val_standalone.py"))
    return p.parse_args()


def signed_area(poly):
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.sum(x*np.roll(y, -1) - y*np.roll(x, -1)))


def polygon_iou_cv2(a, b):
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
        "angle_deg": angle,
        "aspect_ratio": major / max(minor, 1e-12),
    }


def rect_poly(cx, cy, major, minor, angle_deg):
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)], dtype=np.float64)
    v = np.array([-math.sin(t), math.cos(t)], dtype=np.float64)
    c = np.array([cx, cy], dtype=np.float64)
    du = 0.5 * major * u
    dv = 0.5 * minor * v
    return np.stack([c-du-dv, c+du-dv, c+du+dv, c-du+dv], axis=0)


def signed_axis_delta(a, b):
    return float(((b - a + 90.0) % 180.0) - 90.0)


def abs_axis_delta(a, b):
    return abs(signed_axis_delta(a, b))


def parse_gt_file(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            q = line.strip().split()
            if not q:
                continue
            if len(q) < 9:
                rows.append({"malformed": True, "line": line_no})
                continue
            try:
                poly = np.array(list(map(float, q[:8])), dtype=np.float64).reshape(4, 2)
            except Exception:
                rows.append({"malformed": True, "line": line_no})
                continue
            cls = q[8]
            difficult = 0
            if len(q) >= 10:
                try:
                    difficult = int(float(q[9]))
                except Exception:
                    difficult = 0
            rows.append({
                "malformed": False,
                "line": line_no,
                "poly": poly,
                "class": cls,
                "difficult": difficult,
            })
    return rows


def geometry_checks(poly):
    p = np.asarray(poly, dtype=np.float64).reshape(4, 2)
    area = signed_area(p)
    contour = p.astype(np.float32)
    convex = bool(cv2.isContourConvex(contour))
    distinct = len({(round(float(x), 9), round(float(y), 9)) for x, y in p}) == 4

    edges = [p[(i+1) % 4] - p[i] for i in range(4)]
    lens = [float(np.linalg.norm(e)) for e in edges]
    degenerate_edge = any(x <= 1e-9 for x in lens)

    ortho = []
    for i in range(4):
        a = edges[i]
        b = edges[(i+1) % 4]
        den = np.linalg.norm(a) * np.linalg.norm(b)
        ortho.append(abs(float(np.dot(a, b))) / den if den > 0 else 1.0)

    # Opposite-edge parallelism using normalized cross product magnitude.
    parallel = []
    for i, j in ((0, 2), (1, 3)):
        a, b = edges[i], edges[j]
        den = np.linalg.norm(a) * np.linalg.norm(b)
        cross = abs(float(a[0]*b[1] - a[1]*b[0])) / den if den > 0 else 1.0
        parallel.append(cross)

    g = canonical_geom(p)
    rp = rect_poly(g["cx"], g["cy"], g["major"], g["minor"], g["angle_deg"])
    roundtrip_iou = polygon_iou_cv2(p, rp)

    return {
        "signed_area": area,
        "abs_area": abs(area),
        "winding": "CCW" if area > 0 else ("CW" if area < 0 else "ZERO"),
        "convex": convex,
        "four_distinct_points": distinct,
        "degenerate_edge": degenerate_edge,
        "adjacent_nonorthogonality_max": max(ortho),
        "opposite_nonparallelism_max": max(parallel),
        "canonical_roundtrip_iou": roundtrip_iou,
        **g,
    }


def read_detection_file(path):
    out = []
    if not Path(path).is_file():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            q = line.strip().split()
            if not q:
                continue
            if len(q) < 10:
                continue
            poly = np.array(list(map(float, q[2:10])), dtype=np.float64).reshape(4, 2)
            out.append({
                "line": line_no,
                "image_id": q[0],
                "score": float(q[1]),
                "poly": poly,
                "geom": canonical_geom(poly),
            })
    return out


def compare_baseline_refined(baseline_path, refined_path, clamp_csv=None):
    base = read_detection_file(baseline_path)
    ref = read_detection_file(refined_path)
    n = min(len(base), len(ref))
    rows = []
    order_mismatch = 0

    for i in range(n):
        a = base[i]
        b = ref[i]
        if a["image_id"] != b["image_id"]:
            order_mismatch += 1
        ga, gb = a["geom"], b["geom"]
        rows.append({
            "i": i,
            "image_id_base": a["image_id"],
            "image_id_ref": b["image_id"],
            "score_abs_diff": abs(a["score"] - b["score"]),
            "center_shift": math.hypot(ga["cx"]-gb["cx"], ga["cy"]-gb["cy"]),
            "major_rel_diff": abs(ga["major"]-gb["major"]) / max(ga["major"], 1e-12),
            "minor_rel_diff": abs(ga["minor"]-gb["minor"]) / max(ga["minor"], 1e-12),
            "angle_abs_delta_deg": abs_axis_delta(ga["angle_deg"], gb["angle_deg"]),
        })

    df = pd.DataFrame(rows)
    result = {
        "num_baseline": len(base),
        "num_refined": len(ref),
        "num_compared": n,
        "order_image_id_mismatch_count": order_mismatch,
        "score_max_abs_diff": float(df["score_abs_diff"].max()) if len(df) else None,
        "center_shift_max_px": float(df["center_shift"].max()) if len(df) else None,
        "major_rel_diff_max": float(df["major_rel_diff"].max()) if len(df) else None,
        "minor_rel_diff_max": float(df["minor_rel_diff"].max()) if len(df) else None,
        "angle_changed_count_gt_1e6": int((df["angle_abs_delta_deg"] > 1e-6).sum()) if len(df) else 0,
        "angle_delta_median_deg": float(df["angle_abs_delta_deg"].median()) if len(df) else None,
        "angle_delta_max_deg": float(df["angle_abs_delta_deg"].max()) if len(df) else None,
    }

    if clamp_csv and Path(clamp_csv).is_file():
        c = pd.read_csv(clamp_csv)
        raw_col = next((x for x in ["pred_delta_angle_raw_deg", "pred_delta_angle_deg"] if x in c.columns), None)
        used_col = next((x for x in ["pred_delta_angle_used_deg"] if x in c.columns), None)
        if raw_col and used_col:
            expected = c[raw_col].clip(-60.0, 60.0)
            diff = (expected - c[used_col]).abs()
            result["clamp_rows"] = len(c)
            result["clamp_rule_max_abs_error_deg"] = float(diff.max())
            result["num_raw_abs_gt_60"] = int((c[raw_col].abs() > 60.0).sum())
            result["num_used_abs_eq_60"] = int(np.isclose(c[used_col].abs(), 60.0, atol=1e-8).sum())
            result["clamp_rule_exact_within_1e8"] = bool((diff <= 1e-8).all())

    return result, df


def run_evaluator(evaluator, gt_dir, s2_pred, yolo_pred, s2_score, yolo_score):
    cmd = [
        sys.executable, str(evaluator),
        "--gt-dir", str(gt_dir),
        "--s2-pred", str(s2_pred),
        "--yolo-pred", str(yolo_pred),
        "--class-name", "plane",
        "--s2-score", str(s2_score),
        "--yolo-score", str(yolo_score),
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


def make_box(cx, cy, w, h, angle_deg):
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)])
    v = np.array([-math.sin(t), math.cos(t)])
    c = np.array([cx, cy])
    du = 0.5*w*u
    dv = 0.5*h*v
    return np.stack([c-du-dv, c+du-dv, c+du+dv, c-du+dv])


def poly_line(poly, cls="plane", difficult=0):
    return " ".join(f"{x:.6f}" for x in np.asarray(poly).reshape(-1)) + f" {cls} {difficult}\n"


def pred_line(image_id, score, poly):
    return f"{image_id} {score:.6f} " + " ".join(f"{x:.6f}" for x in np.asarray(poly).reshape(-1)) + "\n"


def synthetic_evaluator_tests(evaluator, out):
    syn = out / "synthetic_eval"
    gt = syn / "gt"
    syn.mkdir(exist_ok=True)
    gt.mkdir(exist_ok=True)

    tests = {}

    # Test A: difficult handling
    useful = make_box(100, 100, 80, 30, 20)
    difficult = make_box(250, 100, 80, 30, -30)
    (gt / "SYN1.txt").write_text(
        poly_line(useful, "plane", 0) + poly_line(difficult, "plane", 1),
        encoding="utf-8"
    )
    pred = syn / "pred_difficult.txt"
    pred.write_text(
        pred_line("SYN1", 0.99, difficult) +
        pred_line("SYN1", 0.98, useful),
        encoding="utf-8"
    )
    tests["difficult_handling"] = run_evaluator(evaluator, gt, pred, pred, 0.0, 0.0)

    # Test B: vertex-order invariance
    gt2 = syn / "gt_vertex"
    gt2.mkdir(exist_ok=True)
    p = make_box(120, 130, 100, 25, 37)
    (gt2 / "SYN2.txt").write_text(poly_line(p, "plane", 0), encoding="utf-8")
    cyclic = np.roll(p, 1, axis=0)
    reverse = p[::-1].copy()
    pc = syn / "pred_cyclic.txt"
    pr = syn / "pred_reverse.txt"
    pc.write_text(pred_line("SYN2", 0.9, cyclic), encoding="utf-8")
    pr.write_text(pred_line("SYN2", 0.9, reverse), encoding="utf-8")
    tests["vertex_cyclic"] = run_evaluator(evaluator, gt2, pc, pc, 0.0, 0.0)
    tests["vertex_reverse"] = run_evaluator(evaluator, gt2, pr, pr, 0.0, 0.0)

    # Test C: known oriented IoU threshold behavior
    gt3 = syn / "gt_iou"
    gt3.mkdir(exist_ok=True)
    a = make_box(100, 100, 100, 30, 25)
    # horizontal-axis shift in global x, chosen to yield moderate IoU
    b = make_box(112, 100, 100, 30, 25)
    expected_iou = polygon_iou_cv2(a, b)
    (gt3 / "SYN3.txt").write_text(poly_line(a, "plane", 0), encoding="utf-8")
    pi = syn / "pred_iou.txt"
    pi.write_text(pred_line("SYN3", 0.9, b), encoding="utf-8")
    r = run_evaluator(evaluator, gt3, pi, pi, 0.0, 0.0)
    r["independent_cv2_iou"] = expected_iou
    tests["known_iou"] = r

    return tests


def extract_metric(stdout, label):
    m = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)\s*%", stdout)
    return float(m.group(1)) if m else None


def extract_count_line(stdout, pattern):
    m = re.search(pattern, stdout)
    return tuple(map(int, m.groups())) if m else None


def summarize_synthetic(tests):
    out = {}
    for name, r in tests.items():
        s = r.get("stdout", "")
        out[name] = {
            "returncode": r.get("returncode"),
            "AP50": extract_metric(s, "AP50"),
            "AP75": extract_metric(s, "AP75"),
            "mAP50_95": extract_metric(s, "mAP50-95"),
            "gt_counts": extract_count_line(s, r"GT utiles\s*:\s*(\d+).*?GT difficult\s*:\s*(\d+)"),
            "independent_cv2_iou": r.get("independent_cv2_iou"),
        }
    return out


def source_context(path, patterns, radius=5):
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    hit_idx = []
    regs = [re.compile(p, re.I) for p in patterns]
    for i, line in enumerate(lines):
        if any(rx.search(line) for rx in regs):
            hit_idx.append(i)
    selected = set()
    for i in hit_idx:
        for j in range(max(0, i-radius), min(len(lines), i+radius+1)):
            selected.add(j)
    return [{"line": j+1, "text": lines[j]} for j in sorted(selected)]


def main():
    args = parse_args()
    project = args.project_root.resolve()
    dataset = (project / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root.resolve()
    out = (project / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    evaluator = (project / args.evaluator).resolve() if not args.evaluator.is_absolute() else args.evaluator.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("="*100)
    print("INFO6544 - STEP 1C GEOMETRY / DIFFICULT / EVALUATION PROTOCOL")
    print("="*100)

    # ---------------------------------------------------------
    # 1. GT geometry / difficult / winding audit
    # ---------------------------------------------------------
    geom_rows = []
    split_counts = {}

    for split in SPLITS:
        ann_dir = dataset / split / "annotations"
        counts = defaultdict(int)
        for p in sorted(ann_dir.glob("*.txt")):
            for r in parse_gt_file(p):
                if r.get("malformed"):
                    counts["malformed"] += 1
                    continue
                if r["class"] != "plane":
                    continue
                g = geometry_checks(r["poly"])
                geom_rows.append({
                    "split": split,
                    "image_id": p.stem,
                    "line": r["line"],
                    "difficult": r["difficult"],
                    **g,
                })
                counts["plane_total"] += 1
                counts["difficult"] += int(r["difficult"] != 0)
                counts["useful"] += int(r["difficult"] == 0)
                counts[f"winding_{g['winding']}"] += 1
                counts["nonconvex"] += int(not g["convex"])
                counts["degenerate"] += int(g["degenerate_edge"] or not g["four_distinct_points"] or g["abs_area"] <= 1e-9)

        split_counts[split] = dict(counts)

    geom = pd.DataFrame(geom_rows)
    geom.to_csv(out / "gt_geometry_audit.csv", index=False)

    geometry_summary = {}
    for split in SPLITS:
        g = geom[geom["split"] == split]
        geometry_summary[split] = {
            **split_counts[split],
            "roundtrip_iou_min": float(g["canonical_roundtrip_iou"].min()) if len(g) else None,
            "roundtrip_iou_median": float(g["canonical_roundtrip_iou"].median()) if len(g) else None,
            "roundtrip_iou_lt_0p99": int((g["canonical_roundtrip_iou"] < 0.99).sum()) if len(g) else 0,
            "adjacent_nonorthogonality_max": float(g["adjacent_nonorthogonality_max"].max()) if len(g) else None,
            "opposite_nonparallelism_max": float(g["opposite_nonparallelism_max"].max()) if len(g) else None,
        }

    # ---------------------------------------------------------
    # 2. Stored-angle convention audit on training matched data
    # ---------------------------------------------------------
    angle_audit = {}
    tm = project / "runs/angle_refiner/train_matches.csv"
    if tm.is_file():
        df = pd.read_csv(tm)
        diffs_s2 = []
        diffs_gt = []
        center_s2 = []
        center_gt = []
        major_s2 = []
        minor_s2 = []

        for _, r in df.iterrows():
            pp = np.array([[r[f"pred_x{i}"], r[f"pred_y{i}"]] for i in range(1,5)], dtype=np.float64)
            gp = np.array([[r[f"gt_x{i}"], r[f"gt_y{i}"]] for i in range(1,5)], dtype=np.float64)
            ps = canonical_geom(pp)
            gs = canonical_geom(gp)

            diffs_s2.append(abs_axis_delta(ps["angle_deg"], float(r["s2_angle_deg"])))
            diffs_gt.append(abs_axis_delta(gs["angle_deg"], float(r["gt_angle_deg"])))
            center_s2.append(math.hypot(ps["cx"]-r["s2_cx"], ps["cy"]-r["s2_cy"]))
            center_gt.append(math.hypot(gs["cx"]-r["gt_cx"], gs["cy"]-r["gt_cy"]))
            major_s2.append(abs(ps["major"]-r["s2_major"]))
            minor_s2.append(abs(ps["minor"]-r["s2_minor"]))

        angle_audit = {
            "rows": len(df),
            "s2_angle_abs_delta_deg_max": float(np.max(diffs_s2)),
            "s2_angle_abs_delta_deg_median": float(np.median(diffs_s2)),
            "gt_angle_abs_delta_deg_max": float(np.max(diffs_gt)),
            "gt_angle_abs_delta_deg_median": float(np.median(diffs_gt)),
            "s2_center_recompute_max_px": float(np.max(center_s2)),
            "gt_center_recompute_max_px": float(np.max(center_gt)),
            "s2_major_recompute_max_px": float(np.max(major_s2)),
            "s2_minor_recompute_max_px": float(np.max(minor_s2)),
            "angle_convention_expected": "major-axis angle in [0,180), axis-periodic modulo 180",
        }

    # ---------------------------------------------------------
    # 3. Verify final TEST refinement changes angle only + clamp
    # ---------------------------------------------------------
    baseline_test = project / "modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/test_final/results_after_nms/plane.txt"
    refined_test = project / "runs/angle_refiner/resnet18_angle/test_final_frozen_clamp60/refined_plane_frozen_clamp60.txt"
    pred_csv = project / "runs/angle_refiner/resnet18_angle/test_final_frozen_clamp60/per_detection_predictions.csv"

    refine_integrity, refine_df = compare_baseline_refined(baseline_test, refined_test, pred_csv)
    refine_df.to_csv(out / "baseline_vs_refined_geometry.csv", index=False)

    # ---------------------------------------------------------
    # 4. Evaluator synthetic tests
    # ---------------------------------------------------------
    synthetic = synthetic_evaluator_tests(evaluator, out)
    synthetic_summary = summarize_synthetic(synthetic)

    # Independent interpretation of synthetic checks.
    difficult_stdout = synthetic["difficult_handling"]["stdout"]
    diff_counts = re.search(r"GT utiles\s*:\s*(\d+)\s*\nGT difficult\s*:\s*(\d+)", difficult_stdout)
    difficult_pass = bool(diff_counts and diff_counts.group(1) == "1" and diff_counts.group(2) == "1")

    vertex_pass = (
        synthetic_summary["vertex_cyclic"]["AP50"] == 100.0
        and synthetic_summary["vertex_reverse"]["AP50"] == 100.0
    )

    kiou = synthetic_summary["known_iou"]["independent_cv2_iou"]
    # AP should be 100 below/equal to actual IoU thresholds and 0 above, except numerical boundary.
    known_iou_stdout = synthetic["known_iou"]["stdout"]
    ap_by_thr = {}
    for thr, val in re.findall(r"AP@([0-9.]+):\s*([0-9.]+)\s*%", known_iou_stdout):
        ap_by_thr[float(thr)] = float(val)

    threshold_consistent = True
    if kiou is not None:
        for thr, val in ap_by_thr.items():
            expected_positive = kiou + 1e-6 >= thr
            observed_positive = val > 99.999
            if expected_positive != observed_positive:
                threshold_consistent = False

    evaluator_tests = {
        "difficult_count_test_passed": difficult_pass,
        "vertex_order_cyclic_and_reverse_invariant": vertex_pass,
        "known_iou_independent_cv2": kiou,
        "known_iou_threshold_behavior_consistent": threshold_consistent,
        "synthetic_summary": synthetic_summary,
        "known_iou_ap_by_threshold": ap_by_thr,
    }

    # ---------------------------------------------------------
    # 5. Re-run known S2 VAL baseline using validated evaluator
    # ---------------------------------------------------------
    s2_val = project / "modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/val_eval/epoch_11/results_after_nms/plane.txt"
    val_eval = run_evaluator(
        evaluator,
        dataset / "val/annotations",
        s2_val,
        s2_val,
        0.2259,
        0.2259,
    )
    val_stdout = val_eval["stdout"]
    val_metrics = {
        "returncode": val_eval["returncode"],
        "AP50": extract_metric(val_stdout, "AP50"),
        "AP75": extract_metric(val_stdout, "AP75"),
        "mAP50_95": extract_metric(val_stdout, "mAP50-95"),
        "expected": {"AP50":94.1715, "AP75":56.6924, "mAP50_95":55.0802},
    }
    if all(val_metrics[k] is not None for k in ("AP50","AP75","mAP50_95")):
        val_metrics["max_abs_difference_points"] = max(
            abs(val_metrics["AP50"]-94.1715),
            abs(val_metrics["AP75"]-56.6924),
            abs(val_metrics["mAP50_95"]-55.0802),
        )

    (out / "validated_s2_val_evaluator_output.txt").write_text(val_stdout, encoding="utf-8")
    for name, r in synthetic.items():
        (out / f"synthetic_{name}_evaluator_output.txt").write_text(r["stdout"] + r["stderr"], encoding="utf-8")

    # ---------------------------------------------------------
    # 6. Source evidence: split->tiling chronology and evaluator
    # ---------------------------------------------------------
    prep = project / "modeles_reference/s2anet/tools/prepare_avions_s2anet.py"
    prep_context = source_context(
        prep,
        patterns=[
            r"dataset_avions",
            r"for\s+split",
            r"train",
            r"val",
            r"test",
            r"1024",
            r"overlap",
            r"images",
            r"annotations",
        ],
        radius=3,
    )
    (out / "prepare_avions_s2anet_relevant_lines.txt").write_text(
        "\n".join(f"L{x['line']}: {x['text']}" for x in prep_context),
        encoding="utf-8"
    )

    eval_context = source_context(
        evaluator,
        patterns=[
            r"difficult",
            r"iou",
            r"inter",
            r"union",
            r"polygon",
            r"ap50",
            r"0\.50",
            r"0\.95",
        ],
        radius=2,
    )
    (out / "evaluator_relevant_lines.txt").write_text(
        "\n".join(f"L{x['line']}: {x['text']}" for x in eval_context),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # 7. Full + compact output
    # ---------------------------------------------------------
    full = {
        "geometry_summary": geometry_summary,
        "stored_angle_convention_audit": angle_audit,
        "final_refinement_integrity": refine_integrity,
        "evaluator_synthetic_tests": evaluator_tests,
        "known_s2_val_metric_reproduction": val_metrics,
        "tiling_preparation_source": str(prep),
        "tiling_preparation_relevant_lines_file": str(out / "prepare_avions_s2anet_relevant_lines.txt"),
        "evaluator_relevant_lines_file": str(out / "evaluator_relevant_lines.txt"),
        "notes": [
            "Geometry audit uses an independent OpenCV convex-polygon intersection implementation.",
            "Canonical angle is defined as the major-axis angle modulo 180 degrees.",
            "The synthetic evaluator tests verify difficult counting, cyclic/reversed vertex-order invariance, and IoU-threshold behavior.",
            "No model or evaluation parameter is changed by this audit."
        ],
    }
    (out / "audit_step1c_full.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    lines = []
    lines += ["="*92, "STEP 1C COMPACT SUMMARY", "="*92, ""]
    lines.append("GT GEOMETRY / DIFFICULT:")
    for split in SPLITS:
        g = geometry_summary[split]
        lines.append(
            f"{split.upper()}: total={g.get('plane_total',0)} useful={g.get('useful',0)} "
            f"difficult={g.get('difficult',0)} CW={g.get('winding_CW',0)} "
            f"CCW={g.get('winding_CCW',0)} nonconvex={g.get('nonconvex',0)} "
            f"degenerate={g.get('degenerate',0)} roundtrip_iou_min={g.get('roundtrip_iou_min'):.8f} "
            f"roundtrip_lt_0.99={g.get('roundtrip_iou_lt_0p99',0)}"
        )

    lines += ["", "STORED ANGLE CONVENTION:"]
    for k,v in angle_audit.items():
        lines.append(f"{k}: {v}")

    lines += ["", "FINAL TEST REFINEMENT INTEGRITY:"]
    for k,v in refine_integrity.items():
        lines.append(f"{k}: {v}")

    lines += ["", "EVALUATOR SYNTHETIC TESTS:"]
    lines.append(f"difficult_count_test_passed: {difficult_pass}")
    lines.append(f"vertex_order_cyclic_and_reverse_invariant: {vertex_pass}")
    lines.append(f"known_iou_independent_cv2: {kiou}")
    lines.append(f"known_iou_threshold_behavior_consistent: {threshold_consistent}")

    lines += ["", "KNOWN S2 VAL METRIC REPRODUCTION:"]
    for k,v in val_metrics.items():
        lines.append(f"{k}: {v}")

    lines += ["", "TILING PREPARATION SOURCE EXCERPT FILE:"]
    lines.append(str(out / "prepare_avions_s2anet_relevant_lines.txt"))
    lines.append("EVALUATOR SOURCE EXCERPT FILE:")
    lines.append(str(out / "evaluator_relevant_lines.txt"))

    compact = out / "audit_step1c_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")

    print(f"Compact summary: {compact}")
    print(f"Full JSON      : {out / 'audit_step1c_full.json'}")
    print("READ-ONLY AUDIT COMPLETE.")
    print("="*100)


if __name__ == "__main__":
    main()
