#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Scientific integrity audit, Step 1A

Scope:
  1) Local provenance/inventory of pretrained and trained weights.
  2) Integrity of ORIGINAL image splits TRAIN / VAL / TEST.
  3) Exact duplicate detection across and within splits:
       - raw-file SHA-256
       - decoded-pixel SHA-256 (catches identical pixels with different encoding)
  4) Image-ID overlap and basic annotation-file integrity.
  5) Git/package/environment provenance snapshots.

READ-ONLY AUDIT:
- Does not modify datasets, checkpoints, predictions, or metrics.
- Does not tune anything on TEST.
- Writes only into the audit output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import torchvision
except Exception:
    torchvision = None


SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--dataset-root", type=Path, default=Path("dataset_avions"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/scientific_audit/step1a_provenance_exact_duplicates"),
    )
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_pixels(path: Path) -> Tuple[Optional[str], Optional[Tuple[int, int, int]], Optional[str]]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, "cv2.imread returned None"
    # Hash shape + dtype + exact decoded bytes.
    h = hashlib.sha256()
    h.update(str(img.shape).encode())
    h.update(str(img.dtype).encode())
    h.update(np.ascontiguousarray(img).tobytes())
    shape = tuple(int(x) for x in img.shape)
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)
    return h.hexdigest(), shape, None


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:
        return {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "error": repr(e),
        }


def safe_json_dump(obj: Any, path: Path):
    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return repr(x)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=conv),
        encoding="utf-8",
    )


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def parse_annotation_basic(path: Path) -> Dict[str, Any]:
    total = 0
    plane = 0
    difficult = 0
    malformed = 0
    classes = defaultdict(int)

    if not path.is_file():
        return {
            "exists": False,
            "total_objects": 0,
            "plane_objects": 0,
            "difficult_objects": 0,
            "malformed_lines": 0,
            "classes": {},
        }

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            q = s.split()
            if len(q) < 9:
                malformed += 1
                continue
            try:
                _ = list(map(float, q[:8]))
            except Exception:
                malformed += 1
                continue
            cls = q[8]
            total += 1
            classes[cls] += 1
            if cls == "plane":
                plane += 1
            if len(q) >= 10:
                try:
                    d = int(float(q[9]))
                    if d != 0:
                        difficult += 1
                except Exception:
                    malformed += 1

    return {
        "exists": True,
        "total_objects": total,
        "plane_objects": plane,
        "difficult_objects": difficult,
        "malformed_lines": malformed,
        "classes": dict(classes),
    }


def git_info(repo: Path) -> Dict[str, Any]:
    if not (repo / ".git").exists():
        return {"repo": str(repo), "is_git_repo": False}

    info = {"repo": str(repo), "is_git_repo": True}
    for key, cmd in [
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("remotes", ["git", "remote", "-v"]),
        ("status", ["git", "status", "--short"]),
        ("last_commit", ["git", "log", "-1", "--format=%H%n%ad%n%s", "--date=iso-strict"]),
    ]:
        info[key] = run_cmd(cmd, cwd=repo)
    return info


def inspect_torch_checkpoint(path: Path) -> Dict[str, Any]:
    out = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    if not path.is_file() or torch is None:
        return out

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        out["torch_load_ok"] = True
        out["object_type"] = type(obj).__name__

        if isinstance(obj, dict):
            out["root_keys"] = [str(k) for k in obj.keys()]

            # Keep only provenance-like scalar/dict fields, not tensors.
            for k in (
                "epoch", "best_dev_median_abs_error_deg", "config",
                "train_args", "date", "version", "license", "docs",
                "meta", "optimizer", "updates"
            ):
                if k in obj:
                    v = obj[k]
                    if isinstance(v, (str, int, float, bool, type(None), dict, list, tuple)):
                        try:
                            json.dumps(v, default=str)
                            out[f"field_{k}"] = v
                        except Exception:
                            out[f"field_{k}"] = repr(v)[:5000]

            # State dict summary.
            for key in ("model_state_dict", "state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    sd = obj[key]
                    out["state_dict_key"] = key
                    out["num_state_tensors"] = len(sd)
                    break

            # Ultralytics checkpoints often contain model/ema modules.
            for key in ("model", "ema"):
                if key in obj and obj[key] is not None:
                    try:
                        m = obj[key]
                        out[f"{key}_type"] = type(m).__name__
                        if hasattr(m, "yaml"):
                            out[f"{key}_yaml"] = getattr(m, "yaml")
                        if hasattr(m, "args"):
                            out[f"{key}_args"] = getattr(m, "args")
                    except Exception as e:
                        out[f"{key}_inspect_error"] = repr(e)
        else:
            out["repr"] = repr(obj)[:2000]

    except Exception as e:
        out["torch_load_ok"] = False
        out["torch_load_error"] = repr(e)

    return out


def grep_config_lines(path: Path, patterns: List[str]) -> Dict[str, Any]:
    out = {"path": str(path), "exists": path.is_file(), "matches": []}
    if not path.is_file():
        return out
    rx = re.compile("|".join(re.escape(p) for p in patterns), flags=re.IGNORECASE)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines, 1):
        if rx.search(line):
            out["matches"].append({"line": i, "text": line.rstrip()})
    return out


def find_named_files(search_roots: List[Path], exact_names: List[str], suffixes: Tuple[str, ...]) -> List[Path]:
    found = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue

        # Fast exact-name probes at root.
        for name in exact_names:
            p = root / name
            if p.is_file() and p.resolve() not in seen:
                found.append(p)
                seen.add(p.resolve())

        # Recursive scan, but prune common huge/unrelated dirs.
        for current, dirs, files in os.walk(root):
            dirs[:] = [
                d for d in dirs
                if d not in {
                    ".git", "node_modules", "__pycache__", ".cache",
                    "anaconda3", "miniconda3", ".conda"
                }
            ]
            for fn in files:
                if fn in exact_names or fn.lower().endswith(suffixes):
                    p = Path(current) / fn
                    # Keep only likely model files.
                    low = str(p).lower()
                    if (
                        fn in exact_names
                        or "s2anet" in low
                        or "yolo26" in low
                        or "angle_refiner" in low
                    ):
                        rp = p.resolve()
                        if rp not in seen:
                            found.append(p)
                            seen.add(rp)
    return sorted(found)


def main():
    args = parse_args()
    project = args.project_root.resolve()
    dataset = (project / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root.resolve()
    out = (project / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("INFO6544 - STEP 1A SCIENTIFIC AUDIT")
    print("=" * 100)
    print(f"Project root : {project}")
    print(f"Dataset root : {dataset}")
    print(f"Output       : {out}")
    print("READ-ONLY DATA AUDIT : yes")
    print("=" * 100)

    # ------------------------------------------------------------------
    # A. Split manifest and exact duplicates.
    # ------------------------------------------------------------------
    manifest = []
    split_ids = {}
    split_summary = {}

    print("\n[1/5] Building original-image manifest and exact hashes...")

    for split in SPLITS:
        images_dir = dataset / split / "images"
        ann_dir = dataset / split / "annotations"
        images = list_images(images_dir)
        ids = [p.stem for p in images]
        split_ids[split] = set(ids)

        raw_hashes = defaultdict(list)
        pixel_hashes = defaultdict(list)
        ann_exists = 0
        malformed_total = 0
        plane_total = 0
        difficult_total = 0

        for idx, p in enumerate(images, 1):
            raw = sha256_file(p)
            pix, shape, pix_err = sha256_pixels(p)
            ann = ann_dir / f"{p.stem}.txt"
            ai = parse_annotation_basic(ann)

            if ai["exists"]:
                ann_exists += 1
            malformed_total += ai["malformed_lines"]
            plane_total += ai["plane_objects"]
            difficult_total += ai["difficult_objects"]

            raw_hashes[raw].append(p.stem)
            if pix:
                pixel_hashes[pix].append(p.stem)

            manifest.append({
                "split": split,
                "image_id": p.stem,
                "image_path": str(p),
                "extension": p.suffix.lower(),
                "file_size_bytes": p.stat().st_size,
                "sha256_file": raw,
                "sha256_pixels": pix,
                "height": shape[0] if shape else None,
                "width": shape[1] if shape else None,
                "channels": shape[2] if shape else None,
                "pixel_decode_error": pix_err,
                "annotation_path": str(ann),
                "annotation_exists": ai["exists"],
                "annotation_total_objects": ai["total_objects"],
                "annotation_plane_objects": ai["plane_objects"],
                "annotation_difficult_objects": ai["difficult_objects"],
                "annotation_malformed_lines": ai["malformed_lines"],
                "annotation_classes_json": json.dumps(ai["classes"], sort_keys=True),
            })

            if idx % 50 == 0 or idx == len(images):
                print(f"  {split.upper():5s}: {idx}/{len(images)} images hashed")

        split_summary[split] = {
            "num_images": len(images),
            "num_unique_image_ids": len(set(ids)),
            "num_annotations_found": ann_exists,
            "num_annotations_missing": len(images) - ann_exists,
            "plane_objects_in_parsed_annotations": plane_total,
            "difficult_objects_in_parsed_annotations_all_classes": difficult_total,
            "malformed_annotation_lines": malformed_total,
            "within_split_duplicate_raw_hash_groups": sum(len(v) > 1 for v in raw_hashes.values()),
            "within_split_duplicate_pixel_hash_groups": sum(len(v) > 1 for v in pixel_hashes.values()),
        }

    # Write manifest.
    manifest_csv = out / "original_image_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        if manifest:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    # ID overlaps.
    id_overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(split_ids[a] & split_ids[b])
        id_overlaps[f"{a}_vs_{b}"] = {
            "count": len(common),
            "ids": common,
        }

    # Exact duplicate groups across all splits.
    by_raw = defaultdict(list)
    by_pix = defaultdict(list)
    for r in manifest:
        by_raw[r["sha256_file"]].append((r["split"], r["image_id"], r["image_path"]))
        if r["sha256_pixels"]:
            by_pix[r["sha256_pixels"]].append((r["split"], r["image_id"], r["image_path"]))

    duplicate_rows = []
    group_id = 0
    for method, groups in (("raw_file_sha256", by_raw), ("decoded_pixel_sha256", by_pix)):
        for h, members in groups.items():
            if len(members) <= 1:
                continue
            splits = sorted(set(x[0] for x in members))
            cross_split = len(splits) > 1
            group_id += 1
            for split, image_id, image_path in members:
                duplicate_rows.append({
                    "group_id": group_id,
                    "method": method,
                    "hash": h,
                    "num_members": len(members),
                    "cross_split": cross_split,
                    "splits": "|".join(splits),
                    "split": split,
                    "image_id": image_id,
                    "image_path": image_path,
                })

    dup_csv = out / "exact_duplicate_groups.csv"
    with dup_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "group_id", "method", "hash", "num_members", "cross_split",
            "splits", "split", "image_id", "image_path"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(duplicate_rows)

    cross_raw_groups = sum(
        len(set(x[0] for x in members)) > 1
        for members in by_raw.values()
        if len(members) > 1
    )
    cross_pix_groups = sum(
        len(set(x[0] for x in members)) > 1
        for members in by_pix.values()
        if len(members) > 1
    )

    # ------------------------------------------------------------------
    # B. Local checkpoint provenance.
    # ------------------------------------------------------------------
    print("\n[2/5] Auditing local checkpoint provenance...")

    known = []

    # Refiner checkpoint.
    refiner = project / "runs/angle_refiner/resnet18_angle/best_angle_refiner.pt"
    known.append(("refiner_best", refiner))

    # Torchvision ImageNet ResNet18 used by the refiner.
    resnet18_cache = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    known.append(("resnet18_imagenet_torchvision_cache", resnet18_cache))

    # Likely YOLO files.
    for p in [
        project / "yolo26l.pt",
        project / "modeles_modernes/yolo26_obb/yolo26l.pt",
        project / "modeles_modernes/yolo26_obb/runs/yolo26l_avions/weights/best.pt",
        project / "modeles_modernes/yolo26_obb/runs/yolo26l_avions/weights/last.pt",
    ]:
        if p.exists():
            known.append(("yolo_candidate", p))

    # Likely S2 final checkpoint.
    s2_root = project / "modeles_reference/s2anet"
    for p in [
        s2_root / "work_dirs/s2anet_avions_r50_fpn/epoch_11.pth",
        s2_root / "work_dirs/s2anet_avions_r50_fpn/latest.pth",
    ]:
        if p.exists():
            known.append(("s2_candidate", p))

    # Search project for additional likely model files.
    search_found = find_named_files(
        [project],
        exact_names=[
            "yolo26l.pt",
            "best.pt",
            "epoch_11.pth",
            "resnet18-f37072fd.pth",
        ],
        suffixes=(".pth", ".pt"),
    )

    for p in search_found:
        role = "discovered_model_file"
        if "s2anet" in str(p).lower():
            role = "s2_discovered"
        elif "yolo26" in str(p).lower():
            role = "yolo26_discovered"
        elif "angle_refiner" in str(p).lower():
            role = "refiner_discovered"
        known.append((role, p))

    # De-duplicate.
    seen = set()
    inventory = []
    for role, p in known:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        info = inspect_torch_checkpoint(p)
        info["role"] = role
        inventory.append(info)

    safe_json_dump(inventory, out / "checkpoint_inventory.json")

    # Flatten important checkpoint inventory fields into CSV.
    inv_rows = []
    for x in inventory:
        inv_rows.append({
            "role": x.get("role"),
            "path": x.get("path"),
            "exists": x.get("exists"),
            "size_bytes": x.get("size_bytes"),
            "sha256": x.get("sha256"),
            "torch_load_ok": x.get("torch_load_ok"),
            "object_type": x.get("object_type"),
            "root_keys": "|".join(x.get("root_keys", [])) if isinstance(x.get("root_keys"), list) else None,
            "epoch": x.get("field_epoch"),
            "version": x.get("field_version"),
            "date": x.get("field_date"),
            "torch_load_error": x.get("torch_load_error"),
        })
    with (out / "checkpoint_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(inv_rows[0].keys()) if inv_rows else [
            "role", "path", "exists", "size_bytes", "sha256"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(inv_rows)

    # ------------------------------------------------------------------
    # C. Training-config provenance lines.
    # ------------------------------------------------------------------
    print("\n[3/5] Inspecting training configuration provenance...")

    configs = {
        "s2_aircraft_config": project / "modeles_reference/s2anet/configs/avions/s2anet_r50_fpn_avions.py",
        "refiner_config": project / "runs/angle_refiner/resnet18_angle/train_config.json",
    }

    config_provenance = {
        "s2_aircraft_config": grep_config_lines(
            configs["s2_aircraft_config"],
            [
                "pretrained", "load_from", "resume_from", "checkpoint",
                "train", "val", "test", "ann_file", "img_prefix"
            ],
        ),
        "refiner_train_config": None,
    }

    if configs["refiner_config"].is_file():
        try:
            config_provenance["refiner_train_config"] = json.loads(
                configs["refiner_config"].read_text(encoding="utf-8")
            )
        except Exception as e:
            config_provenance["refiner_train_config"] = {"error": repr(e)}

    safe_json_dump(config_provenance, out / "training_config_provenance.json")

    # ------------------------------------------------------------------
    # D. Git/package provenance.
    # ------------------------------------------------------------------
    print("\n[4/5] Recording Git and package provenance...")

    git = {
        "s2anet_repo": git_info(s2_root),
    }

    # If project itself is git, capture it too.
    if (project / ".git").exists():
        git["project_repo"] = git_info(project)

    safe_json_dump(git, out / "git_provenance.json")

    pkg = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None) if torch else None,
        "torchvision": getattr(torchvision, "__version__", None) if torchvision else None,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda_torch": getattr(torch.version, "cuda", None) if torch else None,
        "cudnn": torch.backends.cudnn.version() if torch and torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else None,
        "pip_show_ultralytics": run_cmd([sys.executable, "-m", "pip", "show", "ultralytics"]),
        "pip_show_torch": run_cmd([sys.executable, "-m", "pip", "show", "torch"]),
        "pip_show_torchvision": run_cmd([sys.executable, "-m", "pip", "show", "torchvision"]),
    }
    safe_json_dump(pkg, out / "environment_and_packages.json")

    # ------------------------------------------------------------------
    # E. Summary.
    # ------------------------------------------------------------------
    print("\n[5/5] Writing audit summary...")

    summary = {
        "audit_step": "1A_provenance_exact_duplicates",
        "read_only": True,
        "project_root": str(project),
        "dataset_root": str(dataset),
        "split_summary": split_summary,
        "image_id_overlaps": id_overlaps,
        "exact_duplicate_summary": {
            "cross_split_raw_sha256_groups": cross_raw_groups,
            "cross_split_decoded_pixel_sha256_groups": cross_pix_groups,
            "duplicate_rows_written": len(duplicate_rows),
        },
        "checkpoint_inventory_count": len(inventory),
        "output_files": [
            str(manifest_csv),
            str(dup_csv),
            str(out / "checkpoint_inventory.json"),
            str(out / "checkpoint_inventory.csv"),
            str(out / "training_config_provenance.json"),
            str(out / "git_provenance.json"),
            str(out / "environment_and_packages.json"),
        ],
        "limitations": [
            "This step detects exact raw-file and exact decoded-pixel duplicates only; perceptual/near-duplicate analysis is deferred to Step 1B.",
            "Checkpoint provenance is based on local files, embedded metadata, configs, hashes and Git/package state; external upstream provenance still requires explicit verification where local metadata is insufficient.",
            "This step does not yet audit angle convention, vertex ordering, oriented IoU implementation, difficult handling in evaluation, or split-before-tiling. Those are reserved for later Step 1 sub-audits."
        ],
    }
    safe_json_dump(summary, out / "audit_step1a_summary.json")

    print("\n" + "=" * 100)
    print("STEP 1A SUMMARY")
    print("=" * 100)
    for split in SPLITS:
        s = split_summary[split]
        print(
            f"{split.upper():5s} | images={s['num_images']:4d} | "
            f"ann_missing={s['num_annotations_missing']:2d} | "
            f"plane_objects={s['plane_objects_in_parsed_annotations']:5d} | "
            f"malformed={s['malformed_annotation_lines']:2d}"
        )

    print("\nImage-ID overlaps:")
    for k, v in id_overlaps.items():
        print(f"  {k}: {v['count']}")

    print("\nExact cross-split duplicate groups:")
    print(f"  raw-file SHA256 : {cross_raw_groups}")
    print(f"  decoded pixels  : {cross_pix_groups}")

    print(f"\nCheckpoint inventory entries: {len(inventory)}")
    for x in inventory:
        if x.get("exists"):
            print(
                f"  [{x.get('role')}] {x.get('path')} | "
                f"{x.get('size_bytes')} bytes | sha256={x.get('sha256')}"
            )

    print("\nOutputs:")
    for p in summary["output_files"] + [str(out / "audit_step1a_summary.json")]:
        print(f"  {p}")

    print("\nREAD-ONLY AUDIT COMPLETE.")
    print("Do not proceed to near-duplicate conclusions until Step 1B.")
    print("=" * 100)


if __name__ == "__main__":
    main()
