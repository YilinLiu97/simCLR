# SimCLR — Firefighting Device Detection

Self-supervised contrastive learning (SimCLR) applied to CAD firefighting system diagrams.

## Environment Setup

```bash
conda env create -f env.yml
conda activate simclr
```

**Requirements** (managed by `env.yml`):
- Python 3.10
- torch 2.11.0+cu128 / torchvision 0.26.0+cu128 (CUDA 12.8, supports Blackwell sm_120)
- scikit-learn, tensorboard, tqdm, pyyaml, pillow, opencv-python-headless

> GPU note: requires NVIDIA driver ≥ 520 (tested on RTX PRO 6000 Blackwell, CUDA 13.0)

---

## Dataset

YOLO-format dataset with **41 classes** of firefighting device symbols extracted from CAD system diagrams.

```
Firefighting/
├── train/images/   (102 images)
├── train/labels/
├── valid/images/
└── test/images/
```

Images are screenshots of AutoCAD fire-alarm/suppression schematic drawings. Labels are YOLO bounding boxes for each device symbol.

---

## Training

```bash
cd SimCLR
conda activate simclr
python run.py
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `-data` | `/mnt/whitsett/yilinliu/simclr/Firefighting` | Dataset root |
| `-dataset-name` | `firefighting` | Dataset (`firefighting` / `stl10` / `cifar10`) |
| `-a` | `resnet18` | Backbone (`resnet18` / `resnet50`) |
| `-b` | `32` | Batch size |
| `--epochs` | `200` | Number of epochs |
| `--lr` | `3e-4` | Learning rate (Adam) |
| `--temperature` | `0.07` | NT-Xent softmax temperature |
| `--out_dim` | `128` | Projection head output dimension |
| `--gpu-index` | `0` | GPU index |
| `--fp16-precision` | off | Enable FP16 mixed precision |

Example with custom args:
```bash
python run.py --epochs 500 -b 64 --temperature 0.05
```

Checkpoints and TensorBoard logs are saved to `runs/<timestamp>/`.

---

## Model Architecture

```
Input image
    └─▶ ResNet18 (backbone, pretrained=False)
            └─▶ MLP Projection Head
                    Linear(512 → 512) → ReLU → Linear(512 → 128)
                        └─▶ 128-dim feature vector
```

Loss: **NT-Xent (InfoNCE)** — for each image, two augmented views are pushed together while all other views in the batch are pushed apart.

---

## Data Augmentation

Applied to each image **twice** (independently) to generate a positive pair:

| # | Transform | Parameters |
|---|-----------|------------|
| 1 | `RandomResizedCrop` | size=96 |
| 2 | `RandomHorizontalFlip` | p=0.5 |
| 3 | `ColorJitter` (p=0.8) | brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2 |
| 4 | `RandomGrayscale` | p=0.2 |
| 5 | `GaussianBlur` | kernel_size=9, σ ∈ [0.1, 2.0] |

> `ColorJitter` hue requires torchvision ≥ 0.19 (fixed overflow bug in `adjust_hue`).

---

## Plotting Results

```bash
LD_LIBRARY_PATH=/mnt/whitsett/yilinliu/miniconda3/envs/simclr/lib:$LD_LIBRARY_PATH \
    python plot_results.py
```

Outputs saved to `plots/`:

| File | Contents |
|------|----------|
| `plots/training_curves.png` | NT-Xent loss, Top-1/Top-5 contrastive accuracy, learning rate — all vs. training step |
| `plots/tsne.png` | t-SNE (2D) of 128-dim features extracted from all training images, colored by dominant device class per image |

> To change the run to plot, edit `RUN_DIR` in `plot_results.py`.

---

## 1-Shot Recognition

Use `one_shot.py` to evaluate the trained backbone in a **1-shot** setting: one randomly chosen crop per symbol class (from the training set) serves as a prototype; every crop in the test set is then classified by cosine similarity to the nearest prototype.

```bash
python one_shot.py
```

No arguments needed — paths and checkpoint are set inside the file. The script prints overall accuracy and a per-class breakdown.

**How it works:**

```
Training images  →  YOLO labels  →  1 random crop per class
                                         ↓ ResNet18 backbone (frozen, 512-dim)
                                      prototype gallery

Test images  →  YOLO labels  →  all crops
                                    ↓ ResNet18 backbone (frozen, 512-dim)
                                 cosine similarity to each prototype
                                    ↓
                                 predicted class
```

The backbone used is the ResNet18 trained by SimCLR (`runs/Jun20_10-53-45_jason/checkpoint_0200.pth.tar`). Features are extracted **before** the projection head (512-dim), which gives better downstream performance than the 128-dim projected output.

**Baseline result** (seed=42, 40-class, 421 test crops): **42.3% top-1 accuracy** vs. 2.5% random chance.

To change the checkpoint, edit `CKPT_PATH` at the top of `one_shot.py`.

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

Logged every 100 steps: `loss`, `acc/top1`, `acc/top5`, `learning_rate`.
