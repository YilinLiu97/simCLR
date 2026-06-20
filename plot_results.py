import argparse
import os
import glob
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sklearn.manifold import TSNE
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.resnet_simclr import ResNetSimCLR

CLASS_NAMES = [
    '24V-power-cord', 'acousto-optic-alarm', 'area-display',
    'bus-isolation-module', 'coded-smoke-detector', 'coded-temperature-detector',
    'dedicated-metal-box-fire-pump', 'dedicated-metal-box-smoke-fan',
    'dedicated-metal-box-supplementary-fan', 'deflation-indicator-light',
    'electrical-fire-monitoring-line', 'emergency-manual-button',
    'explosion-proof-smoke-detector', 'fire-broadcasting-line',
    'fire-equipment-power-monitoring-line', 'fire-fan-manual-control-line',
    'fire-hydrant-button', 'fire-telephone-extension',
    'fire-water-pump-manual-control-line', 'gas-spray-alarm',
    'infrared-camera-basement', 'i-o-module', 'input-module',
    'light-display', 'manual-alarm-button-with-telephone-jack',
    'manual-automatic-switching-device', 'metal-modular-box',
    'smoke-exhaust-valve-280', 'smoke-exhaust-valve-70',
    'smoke-exhaust-valve-70-closed', 'pressure-switch-flow-switch',
    'pressure-switch-gas-extinguisher', 'safety-signal-valve',
    'secondary-fire-shutter-door-control-box', 'security-video-intercom-door',
    'smoke-vent', 'speaker', 'electromagnetic-valve',
    'video-intercom-card-reader', 'voltage-signal-sensor', 'water-flow-indicator'
]

DATA_DIR = '/mnt/whitsett/yilinliu/simclr/Firefighting'
IMG_DIR  = os.path.join(DATA_DIR, 'train/images')
LBL_DIR  = os.path.join(DATA_DIR, 'train/labels')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir',  required=True,
                   help='TensorBoard run directory (contains events file + checkpoint)')
    p.add_argument('--out_dir',  required=True,
                   help='Output directory for plots')
    p.add_argument('--epochs',   type=int, default=200,
                   help='Number of training epochs (to locate checkpoint file)')
    return p.parse_args()


def plot_curves(run_dir, out_dir):
    print('Reading TensorBoard events...')
    ea = EventAccumulator(run_dir)
    ea.Reload()
    available = ea.Tags()['scalars']
    print(f'  Available scalars: {available}')

    def get_scalar(tag):
        events = ea.Scalars(tag)
        return (np.array([e.step for e in events]),
                np.array([e.value for e in events]))

    loss_steps, loss_vals = get_scalar('loss')
    acc1_steps, acc1_vals = get_scalar('acc/top1')
    acc5_steps, acc5_vals = get_scalar('acc/top5')
    lr_steps,   lr_vals   = get_scalar('learning_rate')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(loss_steps, loss_vals, color='steelblue', linewidth=1.5)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('NT-Xent Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(acc1_steps, acc1_vals, label='Top-1', color='coral', marker='o', markersize=5)
    axes[1].plot(acc5_steps, acc5_vals, label='Top-5', color='seagreen', marker='s', markersize=5)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Contrastive Accuracy\n(logged every 100 steps)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lr_steps, lr_vals, color='mediumpurple', linewidth=1.5)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved → {out_path}')


def plot_tsne(run_dir, out_dir, epochs):
    ckpt_path = os.path.join(run_dir, f'checkpoint_{epochs:04d}.pth.tar')
    print(f'\nLoading checkpoint: {ckpt_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNetSimCLR(base_model='resnet18', out_dim=128)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_paths = sorted(
        glob.glob(os.path.join(IMG_DIR, '*.jpg')) +
        glob.glob(os.path.join(IMG_DIR, '*.png'))
    )
    print(f'Extracting features from {len(img_paths)} images...')

    feats, names = [], []
    with torch.no_grad():
        for path in img_paths:
            img  = Image.open(path).convert('RGB')
            x    = transform(img).unsqueeze(0).to(device)
            feat = model(x).squeeze(0).cpu().numpy()
            feats.append(feat)
            names.append(os.path.basename(path))
    feats = np.array(feats)

    def dominant_class(img_name):
        stem     = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(LBL_DIR, stem + '.txt')
        if not os.path.exists(lbl_path):
            return -1
        with open(lbl_path) as f:
            classes = [int(line.split()[0]) for line in f if line.strip()]
        return Counter(classes).most_common(1)[0][0] if classes else -1

    labels = np.array([dominant_class(n) for n in names])
    valid  = labels >= 0
    feats_v, labels_v = feats[valid], labels[valid]
    print(f'  Images with labels: {valid.sum()} / {len(names)}')
    print(f'  Unique classes present: {len(set(labels_v))}')

    print('\nRunning t-SNE...')
    perplexity = min(15, len(feats_v) - 1)
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=42, max_iter=2000, init='pca')
    coords = tsne.fit_transform(feats_v)

    unique_cls = sorted(set(labels_v))
    cmap = matplotlib.colormaps.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(13, 10))
    for i, cls in enumerate(unique_cls):
        mask = labels_v == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=cmap(i % 20),
                   label=CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                   s=90, alpha=0.85, edgecolors='white', linewidth=0.6)

    ax.set_title(f't-SNE of SimCLR Features  (perplexity={perplexity})', fontsize=14)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7, framealpha=0.9, borderpad=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'tsne.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_curves(args.run_dir, args.out_dir)
    plot_tsne(args.run_dir, args.out_dir, args.epochs)
    print(f'\nDone! All plots in {args.out_dir}')
