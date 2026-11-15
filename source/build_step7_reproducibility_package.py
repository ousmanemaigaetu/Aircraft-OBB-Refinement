#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 7
Build a compact reproducibility package for the aircraft OBB project.

The package contains:
- final frozen predictions required to reproduce TEST metrics
- TEST annotations (text only, no images)
- validated summaries / CSV / JSON outputs
- publication figures and source tables
- key evaluation / analysis scripts
- selected qualitative crops
- environment snapshots when available
- SHA-256 manifest
- a reproducibility README

Large model checkpoints and raw image datasets are NOT copied.
They are referenced by path and, when present, hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory."
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/reproducibility/step7"),
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src: Path, dst: Path, required=False, log=None):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if log is not None:
            log.append(("COPIED", str(src), str(dst)))
        return True
    if required:
        raise FileNotFoundError(f"Required file not found: {src}")
    if log is not None:
        log.append(("MISSING_OPTIONAL", str(src), ""))
    return False


def copy_tree_files(src_dir: Path, dst_dir: Path, patterns=("*",), required=False, log=None):
    if not src_dir.is_dir():
        if required:
            raise FileNotFoundError(f"Required directory not found: {src_dir}")
        if log is not None:
            log.append(("MISSING_OPTIONAL_DIR", str(src_dir), ""))
        return 0

    count = 0
    seen = set()
    for pattern in patterns:
        for src in sorted(src_dir.rglob(pattern)):
            if not src.is_file():
                continue
            rel = src.relative_to(src_dir)
            key = str(rel)
            if key in seen:
                continue
            seen.add(key)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
            if log is not None:
                log.append(("COPIED", str(src), str(dst)))
    return count


def run_capture(cmd, cwd=None):
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=60,
        )
        return r.returncode, r.stdout
    except Exception as e:
        return 999, f"{type(e).__name__}: {e}\n"


def write_environment_snapshots(root: Path, env_dir: Path):
    env_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        ("host_python_version.txt", [sys.executable, "--version"]),
        ("nvidia_smi.txt", ["nvidia-smi"]),
        ("conda_info.txt", ["conda", "info"]),
        ("conda_env_list.txt", ["conda", "env", "list"]),
        ("yolo26_from_history.yml", ["conda", "env", "export", "-n", "yolo26", "--from-history"]),
        ("s2anet_from_history.yml", ["conda", "env", "export", "-n", "s2anet", "--from-history"]),
        (
            "yolo26_runtime_versions.txt",
            [
                "conda", "run", "-n", "yolo26", "python", "-c",
                (
                    "import sys,torch,cv2,numpy;"
                    "print('python',sys.version);"
                    "print('torch',torch.__version__);"
                    "print('torch_cuda',torch.version.cuda);"
                    "print('cudnn',torch.backends.cudnn.version());"
                    "print('opencv',cv2.__version__);"
                    "print('numpy',numpy.__version__);"
                    "\ntry:\n import ultralytics; print('ultralytics', ultralytics.__version__)\n"
                    "except Exception as e: print('ultralytics_error',repr(e))"
                )
            ],
        ),
        (
            "s2anet_runtime_versions.txt",
            [
                "conda", "run", "-n", "s2anet", "python", "-c",
                (
                    "import sys,torch;"
                    "print('python',sys.version);"
                    "print('torch',torch.__version__);"
                    "print('torch_cuda',torch.version.cuda);"
                    "\ntry:\n import mmcv; print('mmcv',mmcv.__version__)\n"
                    "except Exception as e: print('mmcv_error',repr(e))"
                    "\ntry:\n import mmdet; print('mmdet',mmdet.__version__)\n"
                    "except Exception as e: print('mmdet_error',repr(e))"
                )
            ],
        ),
    ]

    results = []
    for filename, cmd in commands:
        code, out = run_capture(cmd, cwd=root)
        (env_dir / filename).write_text(out, encoding="utf-8")
        results.append({
            "file": filename,
            "returncode": code,
            "command": cmd,
        })

    (env_dir / "environment_capture_status.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def collect_qualitative_cases(root: Path, dst: Path, log):
    src_dir = root / "runs/qualitative/step5/individual"
    if not src_dir.is_dir():
        log.append(("MISSING_OPTIONAL_DIR", str(src_dir), ""))
        return []

    wanted = [
        "P1179_pred338",
        "P2236_pred913",
        "P1474_pred434",
        "P1854_pred655",
        "P2689_pred978",
        "P0249_pred185",
    ]

    copied = []
    for token in wanted:
        matches = sorted(src_dir.glob(f"*{token}*.png"))
        if not matches:
            log.append(("MISSING_QUALITATIVE_CASE", token, ""))
            continue
        src = matches[0]
        out = dst / src.name
        copy_file(src, out, required=False, log=log)
        copied.append(src.name)
    return copied


def main():
    args = parse_args()
    root = args.project_root.resolve()
    out_root = (root / args.out_root).resolve()

    package = out_root / "ARTICLE_EXPERIMENTAL_DATA"
    zip_base = out_root / "ARTICLE_EXPERIMENTAL_DATA"

    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)

    log = []

    # ------------------------------------------------------------------
    # Critical frozen evaluation inputs
    # ------------------------------------------------------------------
    critical = [
        (
            root / "modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/test_final/results_after_nms/plane.txt",
            package / "frozen_predictions/s2anet_test_plane.txt",
        ),
        (
            root / "runs/angle_refiner/resnet18_angle/test_final_frozen_clamp60/refined_plane_frozen_clamp60.txt",
            package / "frozen_predictions/s2anet_refined_test_plane_clamp60.txt",
        ),
        (
            root / "runs/angle_refiner/resnet18_angle/test_final_frozen_clamp60/per_detection_predictions.csv",
            package / "frozen_predictions/refiner_test_per_detection_predictions.csv",
        ),
        (
            root / "runs/final_test_geometry/step3/step3_test_geometry_per_match.csv",
            package / "validated_outputs/step3/step3_test_geometry_per_match.csv",
        ),
        (
            root / "runs/bootstrap/step4_paired_image_level/step4_paired_bootstrap_compact_summary.txt",
            package / "validated_outputs/step4/step4_paired_bootstrap_compact_summary.txt",
        ),
        (
            root / "runs/publication/step6/step6_manifest.txt",
            package / "validated_outputs/step6/step6_manifest.txt",
        ),
    ]

    for src, dst in critical:
        copy_file(src, dst, required=True, log=log)

    # TEST annotations only, no image redistribution.
    copy_tree_files(
        root / "dataset_avions/test/annotations",
        package / "test_annotations",
        patterns=("*.txt",),
        required=True,
        log=log,
    )

    # ------------------------------------------------------------------
    # Validated analysis outputs
    # ------------------------------------------------------------------
    copy_tree_files(
        root / "runs/computational_characterization/step2b1_yolo_refiner",
        package / "validated_outputs/step2b1_yolo_refiner",
        patterns=("*.txt", "*.json", "*.csv"),
        required=False,
        log=log,
    )
    copy_tree_files(
        root / "runs/computational_characterization/step2b2_s2anet",
        package / "validated_outputs/step2b2_s2anet",
        patterns=("*.txt", "*.json", "*.csv"),
        required=False,
        log=log,
    )
    copy_tree_files(
        root / "runs/final_test_geometry/step3",
        package / "validated_outputs/step3",
        patterns=("*.txt", "*.json", "*.csv"),
        required=False,
        log=log,
    )
    copy_tree_files(
        root / "runs/bootstrap/step4_paired_image_level",
        package / "validated_outputs/step4",
        patterns=("*.txt", "*.json", "*.csv", "*.npz"),
        required=False,
        log=log,
    )
    copy_tree_files(
        root / "runs/qualitative/step5",
        package / "validated_outputs/step5",
        patterns=("*.txt", "*.csv"),
        required=False,
        log=log,
    )

    # Step 6 figures and tables.
    copy_tree_files(
        root / "runs/publication/step6/figures",
        package / "publication_outputs/figures",
        patterns=("*.pdf", "*.svg"),
        required=True,
        log=log,
    )
    copy_tree_files(
        root / "runs/publication/step6/tables",
        package / "publication_outputs/tables",
        patterns=("*.csv",),
        required=True,
        log=log,
    )

    selected_qualitative = collect_qualitative_cases(
        root,
        package / "publication_outputs/qualitative_selected",
        log,
    )

    # ------------------------------------------------------------------
    # Key scripts / configs
    # ------------------------------------------------------------------
    script_candidates = [
        "eval_obb_val_standalone.py",
        "run_obb_angle_refiner.py",
        "analyze_step3_final_test_geometry.py",
        "bootstrap_step4_paired_image_level.py",
        "select_step5_qualitative_cases.py",
        "make_step6_publication_outputs.py",
        "benchmark_step2b1_yolo_refiner.py",
        "benchmark_step2b2_s2anet.py",
        "measure_step2a1_yolo_refiner.py",
        "measure_step2a2_s2anet.py",
    ]

    for name in script_candidates:
        copy_file(
            root / name,
            package / "scripts" / name,
            required=False,
            log=log,
        )

    config_candidates = [
        (
            root / "modeles_reference/s2anet/configs/avions/s2anet_r50_fpn_avions.py",
            package / "configs/s2anet_r50_fpn_avions.py",
        ),
        (
            root / "modeles_modernes/yolo26_obb/run_yolo26_obb_full.py",
            package / "configs/run_yolo26_obb_full.py",
        ),
        (
            root / "modeles_reference/s2anet/tools/prepare_avions_s2anet.py",
            package / "configs/prepare_avions_s2anet.py",
        ),
    ]
    for src, dst in config_candidates:
        copy_file(src, dst, required=False, log=log)

    # Native FLOPs output if present.
    copy_file(
        root / "runs/computational_characterization/step2a2_s2anet/s2anet_native_flops_1024.txt",
        package / "validated_outputs/step2a/s2anet_native_flops_1024.txt",
        required=False,
        log=log,
    )

    # ------------------------------------------------------------------
    # Environment snapshots
    # ------------------------------------------------------------------
    write_environment_snapshots(root, package / "environment")

    # ------------------------------------------------------------------
    # Hashes for external heavyweight artifacts
    # ------------------------------------------------------------------
    heavyweight_candidates = [
        root / "runs/angle_refiner/resnet18_angle/best_angle_refiner.pt",
        root / "modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/epoch_11.pth",
        root / "modeles_modernes/yolo26_obb/runs/yolo26l_avions/original_image_eval/selected_val_best.pt",
        root / "yolo26l.pt",
    ]

    heavy = []
    for p in heavyweight_candidates:
        item = {
            "path": str(p.relative_to(root)) if p.is_absolute() and str(p).startswith(str(root)) else str(p),
            "exists": p.is_file(),
        }
        if p.is_file():
            item["size_bytes"] = p.stat().st_size
            item["sha256"] = sha256(p)
        heavy.append(item)

    (package / "HEAVYWEIGHT_ARTIFACTS.json").write_text(
        json.dumps(heavy, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Canonical validated facts
    # ------------------------------------------------------------------
    facts = {
        "dataset_test": {
            "original_images": 51,
            "useful_gt": 756,
            "difficult_gt": 16,
        },
        "baseline_s2anet_test": {
            "AP50": 94.2568,
            "AP75": 50.9730,
            "mAP50_95": 51.2632,
            "score_0p50": {
                "precision": 96.7787,
                "recall": 91.4021,
                "F1": 94.0136,
                "TP": 691,
                "FP": 23,
                "FN": 65,
            },
        },
        "refined_s2anet_test": {
            "rule": "angle-only residual refinement, clamp delta to +/-60 degrees",
            "AP50": 94.2568,
            "AP75": 62.9206,
            "mAP50_95": 56.1504,
            "gain_vs_baseline": {
                "AP50": 0.0,
                "AP75": 11.9476,
                "mAP50_95": 4.8873,
            },
        },
        "yolo26l_obb_test": {
            "AP50": 96.49297,
            "AP75": 84.44985,
            "mAP50_95": 68.8126,
        },
        "bootstrap": {
            "unit": "original TEST image",
            "n_images": 51,
            "replicates": 5000,
            "seed": 42,
            "AP75_delta_CI95": [6.7330, 16.8576],
            "AP80_delta_CI95": [9.4188, 19.9994],
            "AP85_delta_CI95": [3.0821, 10.7808],
            "mAP50_95_delta_CI95": [3.4998, 6.1574],
        },
        "compute": {
            "s2anet": {
                "params_total": 36398156,
                "GMAC_1024": 171.776,
                "forward_ms_per_patch": 17.061406007049776,
            },
            "yolo26l_obb": {
                "params_final": 27875556,
                "GFLOPs_native_1024": 259.701604352,
                "forward_ms_per_patch": 7.80244785428144,
            },
            "angle_refiner": {
                "params": 11177538,
                "GMAC_224_per_detection": 1.823522816,
                "forward_ms_batch1_per_detection": 0.6742319416999817,
                "batch64_throughput_detection_s": 8072.110102641896,
            },
        },
        "qualitative_selected": selected_qualitative,
    }

    (package / "VALIDATED_FACTS.json").write_text(
        json.dumps(facts, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = """# ARTICLE_EXPERIMENTAL_DATA

## Purpose

This package contains the compact experimental evidence required to reproduce
and audit the final aircraft OBB results used for the academic report and the
future publication.

No detector or refiner is retrained by this package.

## Frozen TEST comparison

S2A-Net:
- AP50: 94.2568
- AP75: 50.9730
- mAP50-95: 51.2632

S2A-Net + residual angle refiner, clamp +/-60 deg:
- AP50: 94.2568
- AP75: 62.9206
- mAP50-95: 56.1504

Gain:
- AP50: +0.0000
- AP75: +11.9476
- mAP50-95: +4.8873

YOLO26l-OBB:
- AP50: 96.49297
- AP75: 84.44985
- mAP50-95: 68.8126

No state-of-the-art claim is made.

## Evaluation convention

The validated standalone evaluator uses:
- original-image predictions after patch reprojection/fusion
- polygon IoU based on OpenCV convex polygons
- 101-point interpolated AP
- IoU thresholds 0.50:0.05:0.95
- difficult GT ignored
- original TEST split kept untouched for tuning

## Bootstrap

Paired bootstrap:
- unit: original TEST image
- N = 51 images
- 5000 replicates
- seed = 42
- identical image resamples for baseline and refined predictions

The limited number of original TEST images must be reported as a limitation.

## Reproducing final metrics

The package includes:
- `test_annotations/`
- `frozen_predictions/s2anet_test_plane.txt`
- `frozen_predictions/s2anet_refined_test_plane_clamp60.txt`
- `scripts/eval_obb_val_standalone.py`

The original evaluator contains a historical guard preventing paths containing
the word `test`. The final experiment previously used byte-identical neutral
aliases verified by SHA-256. Do not alter prediction or annotation contents.

## Large artifacts

Model checkpoints and raw aerial images are intentionally not duplicated in
this compact package. Their project paths, sizes and SHA-256 values are stored
in `HEAVYWEIGHT_ARTIFACTS.json`.

## Publication outputs

`publication_outputs/figures/`
contains vector PDF/SVG quantitative figures.

`publication_outputs/tables/`
contains the exact CSV source data for those figures.

`publication_outputs/qualitative_selected/`
contains the six preselected clean qualitative crops when available.

## Important interpretation notes

- Conditional pass@IoU 0.75 in Step 3 is NOT AP75.
- Raw DOTA polygons are used for IoU/AP.
- Minimum-area rectangles are used only for canonical angle, aspect ratio and
  size descriptors in the geometric diagnostics.
- The refiner changes angle only; center, dimensions and confidence score are
  preserved.
- Computational overhead claims should separate neural inference from image I/O.
"""
    (package / "README_REPRODUCIBILITY.md").write_text(readme, encoding="utf-8")

    # ------------------------------------------------------------------
    # Package SHA-256 manifest
    # ------------------------------------------------------------------
    manifest_lines = []
    for p in sorted(package.rglob("*")):
        if p.is_file():
            rel = p.relative_to(package)
            manifest_lines.append(f"{sha256(p)}  {rel.as_posix()}")

    (package / "MANIFEST_SHA256.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    # Copy log.
    with (package / "COPY_LOG.tsv").open("w", encoding="utf-8") as f:
        f.write("status\tsource\tdestination\n")
        for row in log:
            f.write("\t".join(row) + "\n")

    # Recompute manifest once more so README/manifest/copylog are all captured
    # except the manifest itself, which would be recursive.
    manifest_lines = []
    for p in sorted(package.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256.txt":
            rel = p.relative_to(package)
            manifest_lines.append(f"{sha256(p)}  {rel.as_posix()}")
    (package / "MANIFEST_SHA256.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # ZIP
    # ------------------------------------------------------------------
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()

    shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=str(out_root),
        base_dir="ARTICLE_EXPERIMENTAL_DATA",
    )

    zip_path = zip_base.with_suffix(".zip")
    zip_hash = sha256(zip_path)

    summary = out_root / "step7_reproducibility_summary.txt"
    summary.write_text(
        "\n".join([
            "=" * 100,
            "STEP 7 REPRODUCIBILITY PACKAGE",
            "=" * 100,
            f"package_dir: {package}",
            f"zip: {zip_path}",
            f"zip_size_bytes: {zip_path.stat().st_size}",
            f"zip_sha256: {zip_hash}",
            f"package_file_count: {sum(1 for p in package.rglob('*') if p.is_file())}",
            f"selected_qualitative_count: {len(selected_qualitative)}",
            "",
            "critical frozen predictions: INCLUDED",
            "TEST text annotations: INCLUDED",
            "validated Step 2B/3/4 outputs: INCLUDED",
            "Step 6 vector figures/tables: INCLUDED",
            "key scripts/configs: INCLUDED when present",
            "environment snapshots: ATTEMPTED",
            "large checkpoints/raw images: NOT COPIED, hashes/paths only",
            "",
            "Next step: build HANDOFF_ARTICLE_UPDATE_03.md and final handoff bundle.",
        ]),
        encoding="utf-8",
    )

    print(summary)
    print(zip_path)
    print("DONE")


if __name__ == "__main__":
    main()
