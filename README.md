# Hyperbolic Prototype Classifier for Flood Detection

A hyperbolic prototype network for flood detection in remote sensing imagery, evaluated on the July 2025 Beijing Miyun flood event.

## Overview

The model encodes remote sensing image patches into the Poincaré ball hyperbolic space and performs binary classification (normal / flood) using class prototypes and a contrastive clustering loss.

**Core modules:**

| File | Description |
|------|-------------|
| `encoder.py` | Multi-scale convolutional encoder with channel attention (PatchEncoder) |
| `hyperbolic.py` | Poincaré ball utilities (Möbius addition, hyperbolic distance, prototype computation, etc.) |
| `classifier.py` | Hyperbolic prototype classifier with contrastive clustering loss (HyperbolicClassifier) |
| `dataset.py` | Remote sensing patch dataset, loads `.npy` format data |
| `trainer.py` | Training loop |
| `inference.py` | Sliding-window inference and uncertainty map visualization |
| `config.py` | Centralized hyperparameter configuration |

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
- `final.npy` — full-scene image for inference, shape: `[H, W, C]`

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

## Author

Fengcheng Ji
