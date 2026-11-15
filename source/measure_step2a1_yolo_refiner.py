#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INFO6544 - Step 2A.1
Computational characterization for YOLO26l-OBB and ResNet-18 angle refiner.

Measures:
- total parameters
- trainable parameters
- checkpoint size on disk
- parameter memory in FP32
- theoretical FLOPs when available from the installed framework

Input sizes:
- YOLO26l-OBB: 1024 x 1024
- ResNet-18 refiner: 224 x 224

No latency benchmark is performed here.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("modeles_modernes/yolo26_obb/runs/yolo26l_avions/original_image_eval/selected_val_best.pt"),
    )
    p.add_argument(
        "--refiner-checkpoint",
        type=Path,
        default=Path("runs/angle_refiner/resnet18_angle/best_angle_refiner.pt"),
    )
    p.add_argument(
        "--refiner-script",
        type=Path,
        default=Path("run_obb_angle_refiner.py"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/computational_characterization/step2a1_yolo_refiner"),
    )
    return p.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def parameter_memory_bytes(model):
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def try_thop(model, x):
    try:
        from thop import profile
    except Exception as e:
        return {"available": False, "error": repr(e)}
    try:
        m = model
        m.eval()
        with torch.no_grad():
            macs, params = profile(m, inputs=(x,), verbose=False)
        return {
            "available": True,
            "macs": float(macs),
            "flops_2x_macs": float(2.0 * macs),
            "params_reported": float(params),
        }
    except Exception as e:
        return {"available": True, "error": repr(e)}


def try_fvcore(model, x):
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        return {"available": False, "error": repr(e)}
    try:
        model.eval()
        with torch.no_grad():
            fa = FlopCountAnalysis(model, x)
            total = float(fa.total())
            unsupported = fa.unsupported_ops()
        return {
            "available": True,
            "flops": total,
            "unsupported_ops": {str(k): int(v) for k, v in unsupported.items()},
        }
    except Exception as e:
        return {"available": True, "error": repr(e)}


def build_refiner(train_script: Path, checkpoint: Path):
    mod = load_module(train_script, "angle_refiner_train_module")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    sig = inspect.signature(mod.build_model)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name.lower() == "pretrained":
            kwargs[name] = False
        elif p.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(f"Cannot map build_model argument: {name}")

    obj = mod.build_model(**kwargs)
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, (tuple, list)):
        model = next(x for x in obj if isinstance(x, torch.nn.Module))
    else:
        raise RuntimeError(f"Unexpected build_model return type: {type(obj)}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, ckpt


def main():
    args = parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    yolo_weights = args.yolo_weights
    ref_ckpt = args.refiner_checkpoint
    ref_script = args.refiner_script

    for label, path in [
        ("YOLO weights", yolo_weights),
        ("Refiner checkpoint", ref_ckpt),
        ("Refiner script", ref_script),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path!r}")

    print("YOLO path repr   :", repr(str(yolo_weights)))
    print("Refiner path repr:", repr(str(ref_ckpt)))

    print("=" * 100)
    print("STEP 2A.1 - YOLO26L-OBB + RESNET18 REFINER")
    print("=" * 100)

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------
    from ultralytics import YOLO

    yolo = YOLO(str(yolo_weights))
    yolo_model = yolo.model
    yolo_total, yolo_trainable = count_params(yolo_model)

    yolo_info = {
        "weights": str(yolo_weights),
        "weights_size_bytes": yolo_weights.stat().st_size,
        "input_shape": [1, 3, 1024, 1024],
        "total_parameters": yolo_total,
        "trainable_parameters": yolo_trainable,
        "parameter_memory_bytes_current_dtype": parameter_memory_bytes(yolo_model),
    }

    # Ultralytics native FLOPs helper if available.
    try:
        from ultralytics.utils.torch_utils import get_flops
        yolo_info["ultralytics_get_flops_GFLOPs"] = float(get_flops(yolo_model, imgsz=1024))
        yolo_info["ultralytics_get_flops_note"] = (
            "Framework-native Ultralytics FLOPs estimate at imgsz=1024."
        )
    except Exception as e:
        yolo_info["ultralytics_get_flops_error"] = repr(e)

    # Do not run THOP manually on YOLO unless needed because some Ultralytics
    # models require specialized forward handling. Native get_flops is preferred.

    # ------------------------------------------------------------------
    # Refiner
    # ------------------------------------------------------------------
    refiner, ckpt = build_refiner(ref_script, ref_ckpt)
    ref_total, ref_trainable = count_params(refiner)

    ref_info = {
        "checkpoint": str(ref_ckpt),
        "checkpoint_size_bytes": ref_ckpt.stat().st_size,
        "input_shape": [1, 3, 224, 224],
        "total_parameters": ref_total,
        "trainable_parameters": ref_trainable,
        "parameter_memory_bytes_current_dtype": parameter_memory_bytes(refiner),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_dev_median_abs_error_deg": ckpt.get("best_dev_median_abs_error_deg"),
    }

    x = torch.zeros(1, 3, 224, 224)
    ref_info["thop"] = try_thop(refiner, x)
    ref_info["fvcore"] = try_fvcore(refiner, x)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    result = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_torch": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "yolo26l_obb": yolo_info,
        "resnet18_angle_refiner": ref_info,
        "notes": [
            "YOLO26l-OBB complexity is reported at 1024x1024 because that is the project inference/training patch size.",
            "Refiner complexity is reported at 224x224 because each local oriented crop is resized to 224x224.",
            "Checkpoint file size is not equal to pure model parameter memory because optimizer/EMA/metadata may also be stored.",
            "Combined S2A-Net + refiner complexity is image-dependent because the refiner runs once per retained S2A-Net detection; this will be characterized later using observed detection counts.",
        ],
    }

    (out / "step2a1_full.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    def fmt_num(x):
        return f"{x:,}" if isinstance(x, int) else str(x)

    lines = []
    lines += [
        "=" * 92,
        "STEP 2A.1 COMPACT SUMMARY",
        "=" * 92,
        "",
        "YOLO26L-OBB",
        f"weights_size_bytes: {yolo_info['weights_size_bytes']}",
        f"total_parameters: {fmt_num(yolo_total)}",
        f"trainable_parameters: {fmt_num(yolo_trainable)}",
        f"parameter_memory_bytes: {yolo_info['parameter_memory_bytes_current_dtype']}",
        f"input_shape: {yolo_info['input_shape']}",
        f"ultralytics_get_flops_GFLOPs: {yolo_info.get('ultralytics_get_flops_GFLOPs')}",
        f"ultralytics_get_flops_error: {yolo_info.get('ultralytics_get_flops_error')}",
        "",
        "RESNET18 ANGLE REFINER",
        f"checkpoint_size_bytes: {ref_info['checkpoint_size_bytes']}",
        f"total_parameters: {fmt_num(ref_total)}",
        f"trainable_parameters: {fmt_num(ref_trainable)}",
        f"parameter_memory_bytes: {ref_info['parameter_memory_bytes_current_dtype']}",
        f"input_shape: {ref_info['input_shape']}",
        f"checkpoint_epoch: {ref_info['checkpoint_epoch']}",
        f"THOP: {ref_info['thop']}",
        f"FVCORE: {ref_info['fvcore']}",
        "",
        "FILES",
        str(out / "step2a1_full.json"),
    ]

    compact = out / "step2a1_compact_summary.txt"
    compact.write_text("\n".join(lines), encoding="utf-8")

    print(compact)
    print("DONE")


if __name__ == "__main__":
    main()
