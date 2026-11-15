# Final experimental validation

## Verified

- Frozen TEST predictions: S2A-Net, refined S2A-Net and YOLO26.
- Polygon-based AP metrics and detection counts.
- 5,000 saved paired bootstrap replicates.
- Geometry analysis on 691 matched detections.
- SHA-256 integrity of the four model checkpoints.
- Original training scripts and environment exports.

## Runtime

- GPU: NVIDIA GeForce RTX 4090 (24 GB).
- Driver: 580.126.20.
- S2A-Net: PyTorch 1.9.1+cu111.
- YOLO26 and refiner: PyTorch 2.11.0+cu128.
- S2A-Net upstream commit:
  9c0f0f65b79cc718172717a9d642bd93c07583d2.

## Limitations

Full retraining and independent inference were not rerun.
SAM3 weights must be obtained separately.
Upstream software and datasets retain their original terms.
