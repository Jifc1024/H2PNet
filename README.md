# Hyperbolic Hierarchy-Aware Prototype Network for On-Orbit Earth Surface Anomaly Detection From Single-Satellite Imagery

A lightweight, single-temporal Earth Surface Anomaly (ESA) detection framework based on hyperbolic prototype learning. This repository implements H2PNet and demonstrates its application to the July 2025 Beijing Miyun flood event.

## Overview

H2PNet is designed for on-orbit ESA detection from a single remote sensing image, without requiring temporally-paired pre-/post-event data at inference time.  The key idea is to:
1) Learn a historical steady-state prior of the normal Earth surface from a few recent anomaly-free images, and compress it into a lightweight prototype set in hyperbolic space.
2) Score test pixels by their deviation from the prototype set using hyperbolic distance, leveraging the exponential expansion of the Poincaré ball to nonlinearly amplify the gap between normal and anomalous features.

**Core modules:**

| File | Description |
|------|-------------|
| `encoder.py` | Hierarchical Feature Encoder: multi-scale convolutional encoder with channel attention |
| `hyperbolic.py` | Poincaré ball utilities (Möbius addition, hyperbolic distance, prototype computation, etc.) |
| `classifier.py` | Hyperbolic prototype classifier with the SACC contrastive clustering loss and momentum prototype update |
| `dataset.py` | Remote sensing patch dataset, loads `.npy` format data |
| `trainer.py` | Training loop with feature-queue-based prototype maintenance |
| `inference.py` | Sliding-window inference and anomaly-score / uncertainty map visualization |
| `config.py` | Centralized hyperparameter configuration |

## Application Case: Beijing Miyun Flood  (July 2025)

This repository instantiates H2PNet for the July 2025 Beijing Miyun flood event. The historical steady-state prior is constructed from cloud-free images, and the resulting prototype set is used to detect flood-inundated pixels on a single post-event image — with no temporally-registered reference image required at inference.

## Requirements

```
torch
numpy
scikit-learn
scipy
tqdm
Pillow
matplotlib
```

## Data Preparation

Place the following files in the `trainSample/` directory:

- `trainData.npy` — training patches, shape: `[N, W, H, C]`
- `label.npy` — training labels, shape: `[N, W, H, num_classes]` (one-hot)
- `final.npy` — full-scene image for inference, shape: `[H, W, C]`, which can be found: https://www.kaggle.com/datasets/fengchengji/beijing-miyun

Place the annotation results in the `testLabel/` directory:

- `label.png` — full-scene reference label (grayscale, foreground pixels > 0)

## Quick Start

**Training:**

Uncomment the training line in `main.py`:

```python
save_path = train(model, dataloader, optimizer, scheduler, device, epochs=config.EPOCHS)
```

Then run:

```bash
python main.py
```

**Inference & Evaluation:**

Make sure `models/final.pth` exists and run:

```bash
python main.py
```

After inference, the uncertainty maps (`uncertainty_map.png`, `uncertainty_map0-1.png`) and evaluation metrics (mIoU, F1, Precision, Recall, OA) will be saved and printed.

## Configuration

All hyperparameters are centralized in `config.py`. Modify and re-run to apply changes.

```python
SEED        = 1111
BATCH_SIZE  = 128
EPOCHS      = 300
LR          = 1e-4
EMBED_DIM   = 16
CURVATURE   = 0.5   # Poincaré ball curvature
BIN_THRESH  = 0.6   # Binarization threshold
...
```
The defaults here are tuned for the Miyun-flood instantiation. The settings reported in the paper for the full ESAD benchmark are: patch size 5×5, batch size 128, learning rate 1e-4 (Adam), curvature c = 0.5, distance threshold Δ = 0.5, contrastive coefficient β = 0.1, prototypes updated every 10 iterations with momentum α = 0.9. Adjust as needed for other ESA scenarios.

## Citation
If you find this work useful, please cite:
@article{ji2025h2pnet,
  title   = {Hyperbolic Hierarchy-aware Prototype Network for On-orbit Earth
             Surface Anomaly Detection from Single-satellite Imagery},
  author  = {Ji, Fengcheng and Jia, Kun and Wang, Qiao and Wei, Haishuo and
             Jiang, Zihang and Xu, Can},
  year    = {2025}
}
