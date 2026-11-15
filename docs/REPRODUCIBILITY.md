# Reproducibility and scope

## Verified results

The three methods' frozen TEST metrics were independently recomputed locally using the same polygon evaluator. The S2A-Net and residual-refiner checkpoints, the selected YOLO26 checkpoint and generic YOLO26 initialization were checked against the recorded SHA-256 manifest. The saved paired-bootstrap replicates and fixed matched-geometry results were checked. These checks do not constitute retraining or fresh model inference.

## Recorded runtime

- GPU: NVIDIA GeForce RTX 4090, driver 580.126.20, 24,564 MiB reported.
- S2A-Net: Python 3.7, PyTorch 1.9.1+cu111; local CUDA 11.3 toolchain for legacy extensions. Upstream commit `9c0f0f65b79cc718172717a9d642bd93c07583d2`.
- YOLO26 and refiner: Python 3.11, PyTorch 2.11.0+cu128.
- Full Conda and pip exports are included in the separately collected academic package, not this repository.

## Running the selected scripts

The code is the original experimental code, not a turnkey installation. Create the appropriate environments from the recorded exports if you have the academic package, obtain DOTA data and checkpoints under their terms, and inspect each script's CLI before running. The standalone evaluator's historical CLI guard is VAL-specific; frozen TEST verification in the academic package calls the evaluator's functions directly.

For example, the residual refiner training script exposes its configuration via:

```sh
python source/run_obb_angle_refiner.py --help
```

With the modern environment installed and the original TRAIN images and matched-label CSV accessible, an example matching the recorded training settings is:

```sh
python source/run_obb_angle_refiner.py \
  --matches /path/to/train_matches.csv \
  --images-dir /path/to/TRAIN/images \
  --out-dir /path/to/refiner_run \
  --img-size 224 --crop-scale 2 --min-crop-side 64 \
  --epochs 30 --batch-size 64 --lr 0.0001 --weight-decay 0.0001 \
  --dev-fraction 0.15 --patience 6 --seed 42
```

This example uses CLI arguments verified in the archived script. It has not been rerun in a fresh installation. Some paths, upstream dependencies and data preparation steps are external. The final angle saturation is selected on VAL and frozen before TEST.

The bootstrap CLI accepts the corresponding real evaluation annotations and original-image prediction text files:

```sh
python source/bootstrap_step4_paired_image_level.py \
  --gt-dir /path/to/test_annotations \
  --baseline /path/to/s2anet_test_plane.txt \
  --refined /path/to/s2anet_refined_test_plane_clamp60.txt \
  --out-dir /path/to/bootstrap --replicates 5000 --seed 42
```

The data files and checkpoints are not distributed here. They must be obtained separately under their applicable terms. The original source and documentation should be consulted for SAM3 setup and third-party dependencies. No new licensing grant is implied by this repository.
