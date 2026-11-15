#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OBB ANGLE REFINER - entraînement initial.

But:
- partir d'une détection S2A-Net déjà correcte (IoU >= 0.50 sur TRAIN),
- extraire un crop orienté autour de l'OBB S2A-Net,
- remettre l'axe S2A-Net horizontal dans le crop,
- prédire uniquement le résidu angulaire vers le GT.

Cible périodique:
    y = [cos(2*dtheta), sin(2*dtheta)]
car l'orientation d'une OBB est périodique modulo 180 degrés.

Important:
- TRAIN uniquement
- split interne groupé par image originale
- aucun usage de VAL ou TEST dans ce script
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--crop-scale", type=float, default=2.0)
    p.add_argument("--min-crop-side", type=float, default=64.0)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dev-fraction", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--pretrained",
        choices=["imagenet", "none"],
        default="imagenet"
    )
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_matches(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)

        for row in r:
            rows.append({
                "image_id": row["image_id"],
                "score": float(row["score"]),
                "iou": float(row["iou"]),

                "s2_cx": float(row["s2_cx"]),
                "s2_cy": float(row["s2_cy"]),
                "s2_major": float(row["s2_major"]),
                "s2_minor": float(row["s2_minor"]),
                "s2_angle_deg": float(row["s2_angle_deg"]),

                "gt_angle_deg": float(row["gt_angle_deg"]),
                "gt_aspect_ratio": float(row["gt_aspect_ratio"]),

                "delta_angle_deg": float(row["delta_angle_deg"]),
            })

    return rows


def grouped_split(rows, dev_fraction, seed):
    image_ids = sorted(set(r["image_id"] for r in rows))

    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n_dev = max(
        1,
        int(round(len(image_ids) * dev_fraction))
    )

    dev_ids = set(image_ids[:n_dev])
    train_ids = set(image_ids[n_dev:])

    train_rows = [
        r for r in rows
        if r["image_id"] in train_ids
    ]

    dev_rows = [
        r for r in rows
        if r["image_id"] in dev_ids
    ]

    return train_rows, dev_rows, sorted(train_ids), sorted(dev_ids)


def find_image(images_dir, image_id):
    root = Path(images_dir)

    for ext in [
        ".png", ".jpg", ".jpeg",
        ".tif", ".tiff", ".bmp"
    ]:
        p = root / f"{image_id}{ext}"

        if p.exists():
            return p

    raise FileNotFoundError(
        f"Image introuvable pour {image_id} dans {images_dir}"
    )


def oriented_square_crop(
    image,
    cx,
    cy,
    major,
    angle_deg,
    crop_scale,
    min_crop_side,
    out_size,
):
    """
    Crop carré orienté selon l'angle S2A-Net.

    L'axe principal S2A-Net est envoyé horizontalement dans l'image
    de sortie. La taille du carré est crop_scale * longueur majeure.
    """
    side = max(
        float(min_crop_side),
        float(crop_scale) * float(major)
    )

    theta = math.radians(float(angle_deg))

    u = np.asarray(
        [math.cos(theta), math.sin(theta)],
        dtype=np.float32
    )

    v = np.asarray(
        [-math.sin(theta), math.cos(theta)],
        dtype=np.float32
    )

    c = np.asarray(
        [float(cx), float(cy)],
        dtype=np.float32
    )

    h = 0.5 * side

    # ordre: haut-gauche, haut-droit, bas-droit, bas-gauche
    src = np.stack([
        c - h*u - h*v,
        c + h*u - h*v,
        c + h*u + h*v,
        c - h*u + h*v,
    ]).astype(np.float32)

    s = float(out_size - 1)

    dst = np.asarray([
        [0.0, 0.0],
        [s, 0.0],
        [s, s],
        [0.0, s],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    crop = cv2.warpPerspective(
        image,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return crop


class AngleDataset(Dataset):
    def __init__(
        self,
        rows,
        images_dir,
        img_size,
        crop_scale,
        min_crop_side,
        train,
    ):
        self.rows = rows
        self.images_dir = images_dir
        self.img_size = img_size
        self.crop_scale = crop_scale
        self.min_crop_side = min_crop_side

        aug = []

        if train:
            aug += [
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3)],
                    p=0.15,
                ),
            ]

        aug += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        self.transform = transforms.Compose(aug)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        path = find_image(
            self.images_dir,
            r["image_id"]
        )

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if bgr is None:
            raise RuntimeError(
                f"Impossible de lire {path}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        crop = oriented_square_crop(
            rgb,
            r["s2_cx"],
            r["s2_cy"],
            r["s2_major"],
            r["s2_angle_deg"],
            self.crop_scale,
            self.min_crop_side,
            self.img_size,
        )

        pil = Image.fromarray(crop)
        x = self.transform(pil)

        d = math.radians(
            2.0 * r["delta_angle_deg"]
        )

        target = torch.tensor(
            [math.cos(d), math.sin(d)],
            dtype=torch.float32
        )

        return {
            "image": x,
            "target": target,
            "delta_deg": torch.tensor(
                r["delta_angle_deg"],
                dtype=torch.float32
            ),
            "aspect_ratio": torch.tensor(
                r["gt_aspect_ratio"],
                dtype=torch.float32
            ),
        }


def build_model(pretrained):
    used_pretrained = False

    if pretrained == "imagenet":
        try:
            from torchvision.models import ResNet18_Weights

            model = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )

            used_pretrained = True

        except Exception as e:
            print(
                "WARNING: poids ImageNet indisponibles:",
                repr(e)
            )
            print(
                "Fallback: ResNet18 initialisé aléatoirement."
            )
            model = resnet18(weights=None)

    else:
        try:
            model = resnet18(weights=None)
        except TypeError:
            model = resnet18(pretrained=False)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model, used_pretrained


def normalize_vec(x):
    return x / torch.clamp(
        torch.linalg.norm(
            x,
            dim=1,
            keepdim=True
        ),
        min=1e-8
    )


def periodic_cosine_loss(raw, target):
    pred = normalize_vec(raw)

    return (
        1.0
        - torch.sum(
            pred * target,
            dim=1
        )
    ).mean()


def vector_to_delta_deg(v):
    v = normalize_vec(v)

    angle2 = torch.atan2(
        v[:, 1],
        v[:, 0]
    )

    return torch.rad2deg(
        0.5 * angle2
    )


def signed_axis_delta_tensor(source, target):
    # résultat équivalent modulo 180 dans [-90, 90)
    return torch.remainder(
        target - source + 90.0,
        180.0
    ) - 90.0


def evaluate(model, loader, device):
    model.eval()

    losses = []
    abs_errors = []
    pred_deltas = []
    gt_deltas = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(
                device,
                non_blocking=True
            )

            target = batch["target"].to(
                device,
                non_blocking=True
            )

            gt_delta = batch["delta_deg"].to(
                device,
                non_blocking=True
            )

            raw = model(x)

            loss = periodic_cosine_loss(
                raw,
                target
            )

            pred_delta = vector_to_delta_deg(
                raw
            )

            err = torch.abs(
                signed_axis_delta_tensor(
                    gt_delta,
                    pred_delta
                )
            )

            losses.extend(
                [float(loss.item())] * len(x)
            )

            abs_errors.extend(
                err.detach().cpu().numpy().tolist()
            )

            pred_deltas.extend(
                pred_delta.detach().cpu().numpy().tolist()
            )

            gt_deltas.extend(
                gt_delta.detach().cpu().numpy().tolist()
            )

    e = np.asarray(abs_errors, dtype=float)

    return {
        "loss": float(np.mean(losses)),
        "mae_deg": float(np.mean(e)),
        "median_abs_error_deg": float(np.median(e)),
        "q75_abs_error_deg": float(np.percentile(e, 75)),
        "q90_abs_error_deg": float(np.percentile(e, 90)),
    }


def main():
    args = parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = read_matches(args.matches)

    (
        train_rows,
        dev_rows,
        train_ids,
        dev_ids,
    ) = grouped_split(
        rows,
        args.dev_fraction,
        args.seed
    )

    split = {
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "num_all_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_dev_rows": len(dev_rows),
        "num_train_images": len(train_ids),
        "num_dev_images": len(dev_ids),
        "train_image_ids": train_ids,
        "dev_image_ids": dev_ids,
    }

    with open(
        out_dir / "internal_split.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            split,
            f,
            indent=2,
            ensure_ascii=False
        )

    train_ds = AngleDataset(
        train_rows,
        args.images_dir,
        args.img_size,
        args.crop_scale,
        args.min_crop_side,
        train=True,
    )

    dev_ds = AngleDataset(
        dev_rows,
        args.images_dir,
        args.img_size,
        args.crop_scale,
        args.min_crop_side,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model, used_pretrained = build_model(
        args.pretrained
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version_torch": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "used_imagenet_pretrained": used_pretrained,
        "num_train_samples": len(train_ds),
        "num_dev_samples": len(dev_ds),
        "num_train_images": len(train_ids),
        "num_dev_images": len(dev_ids),
        "target_representation": "[cos(2*dtheta), sin(2*dtheta)]",
        "loss": "periodic cosine loss",
    })

    with open(
        out_dir / "train_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 78)
    print("OBB ANGLE REFINER - TRAIN")
    print("=" * 78)
    print("Samples total       :", len(rows))
    print("Train samples       :", len(train_ds))
    print("Dev samples         :", len(dev_ds))
    print("Train images        :", len(train_ids))
    print("Dev images          :", len(dev_ids))
    print("Device              :", device)
    print("ImageNet pretrained :", used_pretrained)
    print("Crop scale          :", args.crop_scale)
    print("Image size          :", args.img_size)
    print("=" * 78)

    log_rows = []

    best_metric = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        running = []
        batch_count = len(train_loader)

        for bi, batch in enumerate(
            train_loader,
            start=1
        ):
            x = batch["image"].to(
                device,
                non_blocking=True
            )

            target = batch["target"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            raw = model(x)

            loss = periodic_cosine_loss(
                raw,
                target
            )

            loss.backward()
            optimizer.step()

            running.append(
                float(loss.item())
            )

            if (
                bi == 1
                or bi == batch_count
                or bi % 25 == 0
            ):
                print(
                    f"Epoch {epoch:02d}/{args.epochs} "
                    f"batch {bi:03d}/{batch_count} "
                    f"loss={np.mean(running):.5f}",
                    flush=True
                )

        train_loss = float(
            np.mean(running)
        )

        dev = evaluate(
            model,
            dev_loader,
            device
        )

        metric = dev[
            "median_abs_error_deg"
        ]

        scheduler.step(metric)

        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev["loss"],
            "dev_mae_deg": dev["mae_deg"],
            "dev_median_abs_error_deg":
                dev["median_abs_error_deg"],
            "dev_q75_abs_error_deg":
                dev["q75_abs_error_deg"],
            "dev_q90_abs_error_deg":
                dev["q90_abs_error_deg"],
            "lr": lr,
        }

        log_rows.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.5f} | "
            f"dev_med={dev['median_abs_error_deg']:.3f}deg | "
            f"dev_MAE={dev['mae_deg']:.3f}deg | "
            f"q90={dev['q90_abs_error_deg']:.3f}deg | "
            f"lr={lr:.2e}"
        )

        if metric < best_metric - 1e-6:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict":
                    model.state_dict(),
                "epoch": epoch,
                "best_dev_median_abs_error_deg":
                    best_metric,
                "config": config,
            }, out_dir / "best_angle_refiner.pt")

            print(
                "  -> meilleur checkpoint sauvegardé"
            )

        else:
            epochs_without_improvement += 1

        with open(
            out_dir / "training_log.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(
                    log_rows[0].keys()
                )
            )
            w.writeheader()
            w.writerows(log_rows)

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                f"Early stopping après epoch {epoch}; "
                f"meilleur epoch={best_epoch}."
            )
            break

    summary = {
        "best_epoch": best_epoch,
        "best_dev_median_abs_error_deg":
            best_metric,
        "checkpoint":
            str(out_dir / "best_angle_refiner.pt"),
        "num_epochs_ran": len(log_rows),
    }

    with open(
        out_dir / "training_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 78)
    print("ENTRAINEMENT TERMINE")
    print("=" * 78)
    print("Best epoch :", best_epoch)
    print(
        "Best dev median abs error :",
        f"{best_metric:.4f} deg"
    )
    print(
        "Checkpoint :",
        out_dir / "best_angle_refiner.pt"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
