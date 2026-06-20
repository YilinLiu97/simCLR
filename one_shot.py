"""
1-shot recognition on the firefighting CAD dataset.

Support set : 1 randomly chosen crop per class from the training set
Query set   : every annotated crop from the test set
Backbone    : trained SimCLR ResNet18 (frozen), 512-dim features before
              the projection head, L2-normalised
Matching    : cosine similarity to the nearest prototype
"""

import os, sys, glob, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.resnet_simclr import ResNetSimCLR

# ── paths ──────────────────────────────────────────────────────────────────
CKPT_PATH = 'runs/Jun20_10-53-45_jason/checkpoint_0200.pth.tar'
DATA_DIR  = '/mnt/whitsett/yilinliu/simclr/Firefighting'
TRAIN_IMG = os.path.join(DATA_DIR, 'train/images')
TRAIN_LBL = os.path.join(DATA_DIR, 'train/labels')
TEST_IMG  = os.path.join(DATA_DIR, 'test/images')
TEST_LBL  = os.path.join(DATA_DIR, 'test/labels')

SEED = 42
MIN_CROP_PX = 16   # ignore crops smaller than this in either dimension

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


# ── model ──────────────────────────────────────────────────────────────────
def load_backbone(ckpt_path, device):
    model = ResNetSimCLR(base_model='resnet18', out_dim=128)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    # Hook to grab 512-dim features BEFORE the projection head
    _cache = {}
    def _hook(module, inp, out):
        _cache['feat'] = out.flatten(1)   # [B, 512]
    model.backbone.avgpool.register_forward_hook(_hook)

    @torch.no_grad()
    def extract(crop_pil):
        x = transform(crop_pil).unsqueeze(0).to(device)
        model(x)
        feat = _cache['feat'].squeeze(0)
        return F.normalize(feat, dim=0).cpu()

    return extract


transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── YOLO crop helper ───────────────────────────────────────────────────────
def load_crops(img_dir, lbl_dir, min_px=MIN_CROP_PX):
    """Return dict: class_id → list of PIL crops, across all images."""
    result = defaultdict(list)
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')))
    for img_path in img_paths:
        img  = Image.open(img_path).convert('RGB')
        W, H = img.size
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl  = os.path.join(lbl_dir, stem + '.txt')
        if not os.path.exists(lbl):
            continue
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0, int((cx - bw / 2) * W))
                y1 = max(0, int((cy - bh / 2) * H))
                x2 = min(W,  int((cx + bw / 2) * W))
                y2 = min(H,  int((cy + bh / 2) * H))
                if (x2 - x1) >= min_px and (y2 - y1) >= min_px:
                    result[cls].append(img.crop((x1, y1, x2, y2)))
    return result


# ── main ───────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    extract = load_backbone(CKPT_PATH, device)

    # ── 1. Build support set: 1 crop per class from training set ──────────
    print('\nBuilding support set (1 crop/class from training)...')
    train_crops = load_crops(TRAIN_IMG, TRAIN_LBL)
    prototypes  = {}   # class_id → feature tensor [512]
    for cls, crops in sorted(train_crops.items()):
        ref = random.choice(crops)
        prototypes[cls] = extract(ref)
    print(f'  Classes with prototypes : {len(prototypes)}')

    proto_ids  = sorted(prototypes.keys())
    proto_mat  = torch.stack([prototypes[c] for c in proto_ids])  # [K, 512]

    # ── 2. Build query set: all crops from test set ────────────────────────
    print('\nLoading test crops...')
    test_crops = load_crops(TEST_IMG, TEST_LBL)
    n_query    = sum(len(v) for v in test_crops.values())
    n_classes  = len(test_crops)
    print(f'  Test classes  : {n_classes}')
    print(f'  Test crops    : {n_query}')

    # ── 3. 1-shot recognition ─────────────────────────────────────────────
    print('\nRunning 1-shot recognition...')
    correct         = 0
    total           = 0
    class_correct   = defaultdict(int)
    class_total     = defaultdict(int)
    skipped_classes = set()

    for true_cls, crops in sorted(test_crops.items()):
        if true_cls not in prototypes:
            skipped_classes.add(true_cls)
            continue
        for crop in crops:
            feat     = extract(crop)
            sims     = F.cosine_similarity(feat.unsqueeze(0), proto_mat, dim=1)
            pred_cls = proto_ids[sims.argmax().item()]

            class_total[true_cls]   += 1
            if pred_cls == true_cls:
                class_correct[true_cls] += 1
                correct += 1
            total += 1

    # ── 4. Results ────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  1-Shot Accuracy : {correct}/{total} = {100*correct/total:.1f}%')
    print(f'{"="*60}')

    print('\nPer-class accuracy:')
    print(f'  {"Class":<46} {"Correct":>7}  {"Total":>5}  {"Acc":>5}')
    print(f'  {"-"*46}  {"-"*7}  {"-"*5}  {"-"*5}')
    for cls in sorted(class_total.keys()):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
        acc  = 100 * class_correct[cls] / class_total[cls]
        print(f'  {name:<46}  {class_correct[cls]:>7}  {class_total[cls]:>5}  {acc:>4.0f}%')

    if skipped_classes:
        print(f'\n  Skipped (no training prototype): '
              f'{[CLASS_NAMES[c] for c in sorted(skipped_classes)]}')


if __name__ == '__main__':
    main()
