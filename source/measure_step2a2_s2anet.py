#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 2A.2
Computational characterization of the final S2A-Net checkpoint.

Run inside the legacy `s2anet` environment from the project root.

Measures:
- exact total/trainable parameter count of the loaded final architecture
- checkpoint size
- parameter memory
- FLOPs/MACs at 1024x1024 when supported by the installed MMCV/MMDetection stack
- provenance of the config actually used

No latency measurement is performed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo",
        type=Path,
        default=Path("modeles_reference/s2anet"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modeles_reference/s2anet/work_dirs/s2anet_avions_r50_fpn/epoch_11.pth"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config. If omitted, auto-discovery is used.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/computational_characterization/step2a2_s2anet"),
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    return p.parse_args()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def param_memory_bytes(model):
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def discover_config(repo: Path, checkpoint: Path, explicit: Path | None, out: Path):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Explicit config not found: {explicit}")
        return explicit, "explicit_argument"

    work_dir = checkpoint.parent

    candidates = []
    # Most reliable: config copied into the work directory.
    candidates += sorted(work_dir.glob("*.py"))

    # Project-specific configs.
    for pattern in (
        "**/*avions*.py",
        "**/*avion*.py",
        "**/*aircraft*.py",
        "**/s2anet*r50*fpn*.py",
    ):
        candidates += sorted(repo.glob(pattern))

    # Unique candidates while keeping order.
    seen = set()
    unique = []
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique.append(p)

    # Prefer filenames mentioning avions / aircraft.
    for p in unique:
        low = p.name.lower()
        if "avion" in low or "aircraft" in low:
            return p, "auto_discovered_project_config"

    # Checkpoint meta may contain the exact config text.
    try:
        raw = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(raw, dict):
            meta = raw.get("meta", {})
            cfg_text = meta.get("config") if isinstance(meta, dict) else None
            if isinstance(cfg_text, str) and "model" in cfg_text:
                cfg_path = out / "config_recovered_from_checkpoint.py"
                cfg_path.write_text(cfg_text, encoding="utf-8")
                return cfg_path, "checkpoint_meta_config"
    except Exception:
        pass

    # Last resort only if a single plausible S2A-Net R50 FPN config exists.
    plausible = [
        p for p in unique
        if ("s2anet" in p.name.lower() and "r50" in p.name.lower() and "fpn" in p.name.lower())
    ]
    if len(plausible) == 1:
        return plausible[0], "single_plausible_repo_config"

    raise RuntimeError(
        "Could not unambiguously determine the S2A-Net config. "
        "Candidates found:\n" + "\n".join(str(p) for p in unique[:30])
    )


def model_complexity(model, h, w):
    result = {
        "input_shape_chw": [3, h, w],
        "available": False,
        "flops": None,
        "macs_if_1mac_2flops": None,
        "method": None,
        "error": None,
    }

    # MMDetection detectors often implement forward_dummy for FLOP counting.
    if hasattr(model, "forward_dummy"):
        model.forward = model.forward_dummy

    try:
        from mmcv.cnn import get_model_complexity_info
        result["available"] = True
        result["method"] = "mmcv.cnn.get_model_complexity_info"
        flops_str, params_str = get_model_complexity_info(
            model,
            (3, h, w),
            as_strings=True,
            print_per_layer_stat=False,
        )
        result["flops_string"] = flops_str
        result["params_string"] = params_str

        # Also request numeric values if this MMCV version supports it.
        try:
            flops_num, params_num = get_model_complexity_info(
                model,
                (3, h, w),
                as_strings=False,
                print_per_layer_stat=False,
            )
            result["flops"] = float(flops_num)
            result["params_reported"] = float(params_num)
            result["macs_if_1mac_2flops"] = float(flops_num) / 2.0
        except Exception as e:
            result["numeric_error"] = repr(e)

        return result
    except Exception as e:
        result["error"] = repr(e)

    # Fallback to THOP if installed and compatible.
    try:
        from thop import profile
        x = torch.zeros(1, 3, h, w)
        model.eval()
        with torch.no_grad():
            macs, params = profile(model, inputs=(x,), verbose=False)
        return {
            "input_shape_chw": [3, h, w],
            "available": True,
            "method": "THOP fallback",
            "macs": float(macs),
            "flops_2x_macs": float(2.0 * macs),
            "params_reported": float(params),
            "error": None,
        }
    except Exception as e2:
        result["fallback_thop_error"] = repr(e2)
        return result


def main():
    args = parse_args()

    repo = args.repo
    checkpoint = args.checkpoint
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if not repo.is_dir():
        raise FileNotFoundError(f"S2A-Net repo not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Ensure project repo is first in imports.
    sys.path.insert(0, str(repo.resolve()))

    import mmcv
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    import mmdet
    from mmdet.models import build_detector

    config_path, config_source = discover_config(repo, checkpoint, args.config, out)

    cfg = Config.fromfile(str(config_path))

    # Disable training config only if needed for construction? Keep architecture unchanged.
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )

    ckpt = load_checkpoint(model, str(checkpoint), map_location="cpu", strict=False)
    model.eval()

    total, trainable_state = count_params(model)

    # requires_grad in a freshly built training architecture reflects whether
    # the config itself freezes any layers. This is the scientifically useful
    # trainable count, unlike an inference-only EMA object.
    frozen = total - trainable_state

    complexity = model_complexity(model, args.height, args.width)

    result = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_torch": torch.version.cuda,
            "mmcv": getattr(mmcv, "__version__", None),
            "mmdet": getattr(mmdet, "__version__", None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "s2anet": {
            "repo": str(repo),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "config": str(config_path),
            "config_source": config_source,
            "input_shape": [1, 3, args.height, args.width],
            "total_parameters": total,
            "trainable_parameters_from_config_state": trainable_state,
            "frozen_parameters_from_config_state": frozen,
            "parameter_memory_bytes_current_dtype": param_memory_bytes(model),
            "complexity": complexity,
        },
        "checkpoint_meta_keys": sorted(list(ckpt.get("meta", {}).keys()))
            if isinstance(ckpt, dict) and isinstance(ckpt.get("meta"), dict)
            else None,
        "notes": [
            "Parameter count is taken from the actual architecture loaded with epoch_11.pth.",
            "Trainable count is read from the freshly built configured model after checkpoint loading; it reflects config-level freezing.",
            "FLOPs are evaluated for one 1024x1024 patch, matching the project patch inference size.",
            "Combined S2A-Net + refiner cost is not simply one S2A FLOP count plus one refiner FLOP count because the refiner executes once per retained detection.",
        ],
    }

    full = out / "step2a2_full.json"
    full.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    c = complexity
    lines = [
        "=" * 92,
        "STEP 2A.2 S2ANET COMPACT SUMMARY",
        "=" * 92,
        f"config: {config_path}",
        f"config_source: {config_source}",
        f"checkpoint: {checkpoint}",
        f"checkpoint_size_bytes: {checkpoint.stat().st_size}",
        f"input_shape: [1, 3, {args.height}, {args.width}]",
        f"total_parameters: {total}",
        f"trainable_parameters_from_config_state: {trainable_state}",
        f"frozen_parameters_from_config_state: {frozen}",
        f"parameter_memory_bytes: {param_memory_bytes(model)}",
        f"complexity_available: {c.get('available')}",
        f"complexity_method: {c.get('method')}",
        f"flops_string: {c.get('flops_string')}",
        f"flops_numeric: {c.get('flops')}",
        f"macs_if_1mac_2flops: {c.get('macs_if_1mac_2flops')}",
        f"thop_macs: {c.get('macs')}",
        f"thop_flops_2x_macs: {c.get('flops_2x_macs')}",
        f"complexity_error: {c.get('error')}",
        f"fallback_thop_error: {c.get('fallback_thop_error')}",
        "",
        "ENVIRONMENT",
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"mmcv: {getattr(mmcv, '__version__', None)}",
        f"mmdet: {getattr(mmdet, '__version__', None)}",
        "",
        f"full_json: {full}",
    ]

    compact = out / "step2a2_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")
    print(compact)
    print("DONE")


if __name__ == "__main__":
    main()
