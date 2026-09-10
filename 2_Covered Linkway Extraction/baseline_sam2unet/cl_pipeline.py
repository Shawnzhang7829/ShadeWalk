"""
Covered Linkway SAM2-UNet + LoRA Pipeline v2
=============================================
Architecture : SAM2.1 Hiera Large (frozen) + LoRA rank=16 + UNet decoder
Loss         : weighted BCE+IoU (structure_loss) + clDice connectivity loss
New in v2    : JSON→mask generation, per-tile prediction save, mosaic stitching
"""

import os, sys, json, glob, time, math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import cv2
from PIL import Image
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from scipy.ndimage import binary_closing, label as scipy_label
import pandas as pd


# =============================================================================
# 0. JSON → Mask Generation
# =============================================================================

def generate_masks_from_json(
    images_dir: str,
    json_dir: str,
    masks_dir: str,
    splits=('train', 'val'),
    verbose: bool = True,
) -> dict:
    """
    Convert LabelMe-v6 JSON polygon annotations to binary PNG masks.

    - Polygons labeled 'linkway' are filled white (255) on a black canvas.
    - Overlapping polygons are OR-merged (all white).
    - JSON files without any matching image use imageHeight/Width from JSON.
    - Files already present in masks_dir are skipped.

    Returns
    -------
    dict  {split: {'total': N, 'written': N, 'empty': N}}
    """
    os.makedirs(masks_dir, exist_ok=True)
    stats = {}

    for split in splits:
        img_split = os.path.join(images_dir, split)
        jso_split = os.path.join(json_dir,   split)
        msk_split = os.path.join(masks_dir,  split)
        os.makedirs(msk_split, exist_ok=True)

        json_files = sorted(glob.glob(os.path.join(jso_split, '*.json')))
        n_total    = len(json_files)
        n_written  = 0
        n_empty    = 0
        n_skipped  = 0
        report_every = 10 if n_total < 1000 else 100

        if verbose:
            print(f"\n[{split}] {n_total} JSON files  ->  {msk_split}")

        for idx, jpath in enumerate(json_files):
            stem     = Path(jpath).stem
            out_path = os.path.join(msk_split, stem + '.png')

            if os.path.exists(out_path):
                n_skipped += 1
                n_written += 1
                continue

            with open(jpath, 'r', encoding='utf-8') as f:
                ann = json.load(f)

            H = ann.get('imageHeight', 1024)
            W = ann.get('imageWidth',  1024)

            # Try to get size from the actual image
            img_path = os.path.join(img_split, stem + '.png')
            if os.path.exists(img_path):
                with Image.open(img_path) as im:
                    W, H = im.size   # PIL: (width, height)

            canvas = np.zeros((H, W), dtype=np.uint8)
            for shp in ann.get('shapes', []):
                if shp.get('shape_type') != 'polygon':
                    continue
                pts = np.array(shp['points'], dtype=np.float32)
                if len(pts) < 3:
                    continue
                cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2).astype(np.int32)], 255)

            if canvas.max() == 0:
                n_empty += 1

            Image.fromarray(canvas).save(out_path)
            n_written += 1

            if verbose and (idx + 1) % report_every == 0:
                print(f"  [{split}] {idx+1}/{n_total} done...")

        stats[split] = {'total': n_total, 'written': n_written,
                        'empty': n_empty,  'skipped_existing': n_skipped}
        if verbose:
            print(f"  [{split}] Complete: {n_written} masks "
                  f"({n_skipped} already existed, {n_empty} empty/no annotation)")

    return stats


# =============================================================================
# 1. clDice Connectivity Loss
# =============================================================================

class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter: int = 40):
        super().__init__()
        self.num_iter = num_iter

    def _erode(self, x):
        p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def _dilate(self, x):
        return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))

    def _open(self, x):
        return self._dilate(self._erode(x))

    def forward(self, x):
        skel = F.relu(x - self._open(x))
        for _ in range(self.num_iter - 1):
            x    = self._erode(x)
            skel = skel + F.relu(x - self._open(x))
        return skel


_soft_skel = SoftSkeletonize(num_iter=40)


def clDice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits)
    sp   = _soft_skel(pred)
    st   = _soft_skel(target)
    tprec = (sp * target).sum() / (sp.sum() + 1e-6)
    tsens = (st * pred).sum()   / (st.sum() + 1e-6)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-6)


# =============================================================================
# 2. LoRA
# =============================================================================

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank
        in_f, out_f  = linear.in_features, linear.out_features
        self.lora_A  = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B  = nn.Parameter(torch.zeros(out_f, rank))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> int:
    """Inject LoRA into all attn.qkv Linear layers in Hiera backbone."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, rank=rank, alpha=alpha)
            count += 1
    return count


def count_trainable_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"Total params: {total/1e6:.1f}M | "
            f"Trainable: {train/1e6:.1f}M ({100*train/total:.1f}%)")


# =============================================================================
# 3. UNet Decoder Blocks
# =============================================================================

class _Conv2dBnRelu(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, dilation=d, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class RFB(nn.Module):
    """Receptive Field Block for multi-scale context aggregation."""
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 4
        self.b0 = _Conv2dBnRelu(in_c, mid, 1)
        self.b1 = nn.Sequential(_Conv2dBnRelu(in_c, mid, 1),
                                 _Conv2dBnRelu(mid,  mid, 3, p=1))
        self.b2 = nn.Sequential(_Conv2dBnRelu(in_c, mid, 1),
                                 _Conv2dBnRelu(mid,  mid, 3, p=3, d=3))
        self.b3 = nn.Sequential(_Conv2dBnRelu(in_c, mid, 1),
                                 _Conv2dBnRelu(mid,  mid, 3, p=5, d=5))
        self.cat = _Conv2dBnRelu(4 * mid, out_c, 3, p=1)
        self.sc  = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        f = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1)
        return F.relu(self.cat(f) + self.sc(x), inplace=True)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c // 2, 2, 2)
        self.conv = DoubleConv(in_c // 2 + skip_c, out_c)

    def forward(self, x, skip):
        x  = self.up(x)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x, skip], dim=1))


# =============================================================================
# 4. SAM2-UNet + LoRA Model
# =============================================================================

class SAM2UNetLoRA(nn.Module):
    """
    SAM2.1 Hiera Large backbone (frozen) + LoRA + UNet decoder.

    Backbone feature channels for 1024x1024 input:
      f0 (32x32)  : 1152 ch   <- bottleneck (coarsest)
      f1 (64x64)  :  576 ch
      f2 (128x128):  288 ch
      f3 (256x256):  144 ch   <- finest

    Decoder output: 4 logit maps upsampled to HxW (coarse→fine).
    """
    FEAT_CH = [1152, 576, 288, 144]   # coarse→fine
    DEC_CH  = [256,  128,  64,  32]

    def __init__(
        self,
        sam2_repo: str,
        sam2_checkpoint: str,
        sam2_config: str  = "configs/sam2.1/sam2.1_hiera_l.yaml",
        lora_rank: int    = 16,
        lora_alpha: float = 32.0,
    ):
        super().__init__()

        # ── Load SAM2 backbone ──────────────────────────────────────────────
        if sam2_repo not in sys.path:
            sys.path.insert(0, sam2_repo)

        # sam2/__init__.py calls initialize_config_module("sam2") on first import.
        # On repeated calls (e.g. notebook reload), GlobalHydra is already
        # initialized → clear it and re-initialize before building.
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_module
        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")

        from sam2.build_sam import build_sam2
        sam2 = build_sam2(sam2_config, sam2_checkpoint, device='cpu')
        self.encoder = sam2.image_encoder

        # Freeze backbone
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Inject LoRA
        n_lora = inject_lora(self.encoder, rank=lora_rank, alpha=lora_alpha)
        print(f"  LoRA injected into {n_lora} QKV layers (rank={lora_rank})")

        # ── UNet Decoder ────────────────────────────────────────────────────
        C, D = self.FEAT_CH, self.DEC_CH
        self.rfb3 = RFB(C[0], D[0])         # 32x32  -> 32x32
        self.up3  = Up(D[0], C[1], D[1])    # 32x32  -> 64x64
        self.up2  = Up(D[1], C[2], D[2])    # 64x64  -> 128x128
        self.up1  = Up(D[2], C[3], D[3])    # 128x128-> 256x256

        # Output heads (multi-scale supervision)
        self.out3 = nn.Conv2d(D[0], 1, 1)   # coarsest
        self.out2 = nn.Conv2d(D[1], 1, 1)
        self.out1 = nn.Conv2d(D[2], 1, 1)
        self.out0 = nn.Conv2d(D[3], 1, 1)   # finest  -> upsampled to H×W

    def _encode(self, x):
        """Run Hiera trunk and return [coarsest→finest] feature list.

        trunk returns 4 tensors finest→coarsest:
          [0] 144ch @ 256x256   [1] 288ch @ 128x128
          [2] 576ch @  64x64    [3] 1152ch @  32x32
        We reverse to coarse→fine to match FEAT_CH = [1152,576,288,144].
        """
        feats = self.encoder.trunk(x)   # list of 4 tensors
        return list(reversed(feats))    # [1152@32, 576@64, 288@128, 144@256]

    def forward(self, x):
        _, _, H, W = x.shape
        f0, f1, f2, f3 = self._encode(x)   # coarse→fine

        d3 = self.rfb3(f0)        # [B, 256, 32, 32]
        d2 = self.up3(d3, f1)     # [B, 128, 64, 64]
        d1 = self.up2(d2, f2)     # [B,  64,128,128]
        d0 = self.up1(d1, f3)     # [B,  32,256,256]

        def _up(t):
            return F.interpolate(t, (H, W), mode='bilinear', align_corners=False)

        return _up(self.out3(d3)), _up(self.out2(d2)), \
               _up(self.out1(d1)), _up(self.out0(d0))   # coarse→fine


# =============================================================================
# 5. Vegetation / Canopy Mask
# =============================================================================

def compute_canopy_mask_np(img_rgb01: np.ndarray,
                           exg_thresh: float = 0.12,
                           close_r: int      = 12) -> np.ndarray:
    """
    Excess-Green vegetation index on a uint8 / float32 [0-1] RGB tile.

    ExG = 2G - R - B  ∈ [-2, 2]
    Dense canopy: ExG > exg_thresh  (tuned for Singapore Google-Maps aerial tiles)

    Returns binary uint8 mask (1 = dense canopy, 0 = visible ground)
    """
    img = img_rgb01.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2.0 * G - R - B
    canopy = (exg > exg_thresh).astype(np.uint8)
    # Morphological close: fill small gaps within canopy patches
    if close_r > 0:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1)
        )
        canopy = cv2.morphologyEx(canopy, cv2.MORPH_CLOSE, kern)
    return canopy          # H×W uint8


def canopy_weight_tensor(imgs_norm: torch.Tensor,
                         canopy_weight: float = 0.25) -> torch.Tensor:
    """
    Derive per-pixel loss weight from a normalised image batch.
    Pixels classified as dense canopy get weight=canopy_weight (default 0.25)
    instead of 1.0 — reducing their gradient contribution during training.

    imgs_norm : [B, 3, H, W] ImageNet-normalised float32
    Returns   : [B, 1, H, W] weight map in [canopy_weight, 1.0]
    """
    _MEAN_T = torch.tensor([0.485, 0.456, 0.406],
                           device=imgs_norm.device).view(1, 3, 1, 1)
    _STD_T  = torch.tensor([0.229, 0.224, 0.225],
                           device=imgs_norm.device).view(1, 3, 1, 1)
    img01 = imgs_norm * _STD_T + _MEAN_T          # de-normalise → [0,1]
    R, G, B = img01[:, 0:1], img01[:, 1:2], img01[:, 2:3]
    exg     = 2.0 * G - R - B                     # [B,1,H,W]
    is_canopy = (exg > 0.12).float()              # 1 = canopy
    # Simple average-pool "closing" to fill small gaps
    is_canopy = F.avg_pool2d(is_canopy, 25, stride=1, padding=12)
    is_canopy = (is_canopy > 0.3).float()
    weight = 1.0 - (1.0 - canopy_weight) * is_canopy   # [canopy_w, 1.0]
    return weight                                  # [B,1,H,W]


# =============================================================================
# 6. Losses
# =============================================================================

def structure_loss(pred: torch.Tensor, mask: torch.Tensor,
                   pos_weight: float = 50.0,
                   pixel_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Weighted BCE + Weighted IoU.

    pixel_weight : optional [B,1,H,W] per-pixel weight (e.g. canopy downweight).
                   If None, uniform weight=1.
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    if pixel_weight is not None:
        weit = weit * pixel_weight          # canopy pixels get lower weight

    pw   = torch.tensor(pos_weight, device=pred.device, dtype=pred.dtype)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none', pos_weight=pw)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-6)

    pred_s = torch.sigmoid(pred)
    inter  = ((pred_s * mask) * weit).sum(dim=(2, 3))
    union  = ((pred_s + mask) * weit).sum(dim=(2, 3))
    wiou   = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def combined_loss(outputs, mask, pos_weight=50.0, cldice_w=0.3,
                  pixel_weight=None):
    """Multi-scale structure loss (4 heads) + clDice on finest output."""
    loss = 0.0
    scale_weights = [0.5, 0.5, 0.5, 1.0]
    for sw, out in zip(scale_weights, outputs):
        loss += sw * structure_loss(out, mask, pos_weight, pixel_weight)
    loss += cldice_w * clDice_loss(outputs[-1], mask)
    return loss


# =============================================================================
# 6. Dataset
# =============================================================================

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _random_shadow_aug(img_01: np.ndarray, prob: float = 0.55) -> np.ndarray:
    """
    Random shadow overlay on a float32 HxWx3 image in [0, 1].

    Simulates the three main shadow sources encountered in Singapore
    aerial/street-level imagery of covered linkways:

      'strip'    — long rectangular shadow from an adjacent building wall
      'wedge'    — trapezoidal shadow (building corner / angled wall)
      'gradient' — soft gradient shadow (sun angle / broad tree canopy)

    Shadow intensity is randomised between 30 % and 65 % darkening so the
    model learns to detect linkways across a wide range of lighting conditions.

    Only applied to the image; the segmentation mask is unchanged.
    """
    if np.random.rand() > prob:
        return img_01

    H, W = img_01.shape[:2]
    alpha = np.zeros((H, W), dtype=np.float32)   # 0 = no shadow, 1 = full shadow

    shadow_type = np.random.choice(['strip', 'wedge', 'gradient'],
                                   p=[0.40, 0.35, 0.25])

    if shadow_type == 'strip':
        # Narrow to medium-width strip — building wall / roof overhang
        horizontal = np.random.rand() > 0.5
        if horizontal:
            r0 = np.random.randint(0, H)
            rw = np.random.randint(H // 10, H // 3)
            alpha[r0: min(r0 + rw, H), :] = 1.0
        else:
            c0 = np.random.randint(0, W)
            cw = np.random.randint(W // 10, W // 3)
            alpha[:, c0: min(c0 + cw, W)] = 1.0

    elif shadow_type == 'wedge':
        # Trapezoid / irregular quadrilateral — angled building edge
        xs = sorted(np.random.randint(0, W, 2).tolist())
        ys = sorted(np.random.randint(0, H, 2).tolist())
        # Add a slight skew so it isn't a perfect rectangle
        skew = np.random.randint(-W // 6, W // 6)
        pts  = np.array([
            [xs[0],          ys[0]],
            [xs[1],          ys[0]],
            [xs[1] + skew,   ys[1]],
            [xs[0] + skew,   ys[1]],
        ], dtype=np.int32)
        cv2.fillPoly(alpha, [pts], 1.0)

    else:  # gradient — smooth brightness falloff across the tile
        t = np.linspace(0.0, 1.0, W if np.random.rand() > 0.5 else H,
                        dtype=np.float32)
        if np.random.rand() > 0.5:
            t = t[::-1]                          # flip direction
        if t.shape[0] == W:
            alpha = np.tile(t, (H, 1))           # horizontal gradient
        else:
            alpha = np.tile(t.reshape(-1, 1), (1, W))  # vertical gradient

    # Soft Gaussian blur on the shadow edge (realistic penumbra)
    blur_r = np.random.randint(15, 60)
    alpha  = cv2.GaussianBlur(alpha, (0, 0), blur_r)

    # Darken by a random factor
    darkness = np.random.uniform(0.30, 0.65)
    shadow   = alpha[:, :, np.newaxis] * darkness   # broadcast to 3 ch
    img_out  = np.clip(img_01 * (1.0 - shadow), 0.0, 1.0)
    return img_out.astype(np.float32)


class LinkwayDataset(Dataset):
    """
    Loads paired image/mask PNGs.
    images_dir/{split}/*.png  and  masks_dir/{split}/*.png
    Only tiles that have BOTH an image AND a mask are included.
    """
    def __init__(self, images_dir, masks_dir, split='train', img_size=1024):
        self.split    = split
        self.img_size = img_size
        img_dir = os.path.join(images_dir, split)
        msk_dir = os.path.join(masks_dir,  split)

        self.samples = []
        for ip in sorted(glob.glob(os.path.join(img_dir, '*.png'))):
            stem = Path(ip).stem
            mp   = os.path.join(msk_dir, stem + '.png')
            if os.path.exists(mp):
                self.samples.append((ip, mp))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        img = np.array(Image.open(ip).convert('RGB'), dtype=np.float32) / 255.0
        msk = np.array(Image.open(mp).convert('L'),   dtype=np.float32)
        msk = (msk > 128).astype(np.float32)

        sz = self.img_size
        if img.shape[0] != sz or img.shape[1] != sz:
            img = cv2.resize(img, (sz, sz))
            msk = cv2.resize(msk, (sz, sz), interpolation=cv2.INTER_NEAREST)

        if self.split == 'train':
            # ── Geometric (applied to both img & mask) ───────────────────
            # Horizontal / vertical flip
            if np.random.rand() > 0.5:
                img = img[:, ::-1].copy();  msk = msk[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[::-1].copy();     msk = msk[::-1].copy()
            # 90° rotation
            k = np.random.choice([0, 1, 2, 3])
            if k:
                img = np.rot90(img, k).copy()
                msk = np.rot90(msk, k).copy()
            # Random scale-crop: zoom in 80-100%, then pad back to sz
            if np.random.rand() > 0.5:
                scale  = np.random.uniform(0.80, 1.0)
                crop_h = int(sz * scale);  crop_w = int(sz * scale)
                r0 = np.random.randint(0, sz - crop_h + 1)
                c0 = np.random.randint(0, sz - crop_w + 1)
                img = cv2.resize(img[r0:r0+crop_h, c0:c0+crop_w], (sz, sz))
                msk = cv2.resize(msk[r0:r0+crop_h, c0:c0+crop_w], (sz, sz),
                                 interpolation=cv2.INTER_NEAREST)

            # ── Photometric (img only, before normalisation) ─────────────
            # Brightness shift
            if np.random.rand() > 0.5:
                img = np.clip(img + np.random.uniform(-0.15, 0.15), 0.0, 1.0)
            # Contrast scale
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.80, 1.20)
                mean   = img.mean(axis=(0, 1), keepdims=True)
                img    = np.clip((img - mean) * factor + mean, 0.0, 1.0)
            # Channel-wise hue/saturation-like jitter
            if np.random.rand() > 0.5:
                img = np.clip(
                    img * np.random.uniform(0.85, 1.15, size=(1, 1, 3)).astype(np.float32),
                    0.0, 1.0
                )
            # Gaussian blur (simulate varying GSD / focus)
            if np.random.rand() > 0.35:
                sigma = np.random.uniform(0.3, 1.5)
                img   = cv2.GaussianBlur(img, (0, 0), sigma)
            # ── Shadow augmentation ───────────────────────────────────────
            # Simulates building / structure shadows that create brightness
            # discontinuities along a connected covered linkway.
            img = _random_shadow_aug(img, prob=0.55)

        # Normalise (ImageNet stats, after augmentation)
        img = (img - _MEAN) / _STD

        return (torch.from_numpy(img.transpose(2, 0, 1).copy()),  # [3, H, W]
                torch.from_numpy(msk).unsqueeze(0))                # [1, H, W]


def get_dataloaders(images_dir, masks_dir, batch_size=2,
                    img_size=1024, num_workers=0):
    tr = LinkwayDataset(images_dir, masks_dir, 'train', img_size)
    va = LinkwayDataset(images_dir, masks_dir, 'val',   img_size)
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=1,          shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va


# =============================================================================
# 7. Training Loop
# =============================================================================

def _train_epoch(model, loader, optimizer, pos_weight, cldice_w, device,
                 epoch, epochs, scaler=None, canopy_weight=0.25):
    """
    One training epoch with canopy-aware loss weighting.

    canopy_weight : loss weight for pixels under dense tree canopy (default 0.25).
                    Set to 1.0 to disable canopy downweighting.
    """
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f"  Epoch {epoch:>3}/{epochs}",
               leave=False, dynamic_ncols=True, unit="batch")
    for imgs, msks in bar:
        imgs, msks = imgs.to(device), msks.to(device)
        # Per-pixel canopy weight (computed from de-normalised image, no grad)
        with torch.no_grad():
            pw_map = canopy_weight_tensor(imgs, canopy_weight=canopy_weight)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = combined_loss(outputs, msks, pos_weight=pos_weight,
                                     cldice_w=cldice_w, pixel_weight=pw_map)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = combined_loss(outputs, msks, pos_weight=pos_weight,
                                 cldice_w=cldice_w, pixel_weight=pw_map)
            loss.backward()
            optimizer.step()
        total += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def _validate(model, loader, device, threshold=0.5):
    model.eval()
    dice, iou = 0.0, 0.0
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        pred = (torch.sigmoid(model(imgs)[-1]) > threshold).float()
        tp = (pred * msks).sum().item()
        fp = (pred * (1 - msks)).sum().item()
        fn = ((1 - pred) * msks).sum().item()
        dice += (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou  += tp / (tp + fp + fn + 1e-7)
    n = max(len(loader), 1)
    return dice / n, iou / n


def train(
    model,
    train_loader,
    val_loader,
    epochs: int           = 50,
    lr: float             = 1e-4,
    weight_decay: float   = 5e-4,
    cldice_weight: float  = 0.3,
    pos_weight: float     = 50.0,
    save_dir: str         = None,
    device: str           = "cuda",
    use_amp: bool         = True,
    early_stop_patience: int = 15,
    canopy_weight: float  = 0.25,
    verbose: bool         = True,
) -> str:
    """
    Train SAM2UNetLoRA model with AMP, cosine LR decay, and early stopping.

    Parameters
    ----------
    use_amp              : mixed precision (~2x speedup on Ada/Ampere GPUs)
    early_stop_patience  : stop if val_dice does not improve for this many
                           consecutive epochs (0 = disabled)

    Returns path to best checkpoint saved so far (or None).
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device == 'cuda') else None
    if verbose and scaler is not None:
        print("  AMP (mixed precision) enabled — GradScaler active")
    if verbose:
        print(f"  Canopy downweight: {canopy_weight}  "
              f"(tree-occluded pixels weighted at {canopy_weight:.0%} in loss)")

    best_dice, best_path = -1.0, None
    no_improve = 0  # early-stopping counter

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = _train_epoch(model, train_loader, optimizer,
                               pos_weight, cldice_weight, device,
                               epoch, epochs, scaler=scaler,
                               canopy_weight=canopy_weight)
        val_dice, val_iou = _validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        saved = ""
        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            if save_dir:
                best_path = os.path.join(
                    save_dir, f"best_model_dice{val_dice:.4f}.pth"
                )
                torch.save(model.state_dict(), best_path)
                saved = " [SAVED]"
        else:
            no_improve += 1

        if verbose:
            es_info = (f"  [no-improve {no_improve}/{early_stop_patience}]"
                       if early_stop_patience > 0 and no_improve > 0 else "")
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"loss={tr_loss:.4f} | "
                  f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | "
                  f"{elapsed:.0f}s{saved}{es_info}")

        # Early stopping
        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            if verbose:
                print(f"\n[Early Stop] val_dice did not improve for "
                      f"{early_stop_patience} epochs. Best={best_dice:.4f}")
            break

    return best_path


# =============================================================================
# 8. Tile Inference on Full GeoTIF (sliding window)
# =============================================================================

def _preprocess_rgb(rgb_hw3: np.ndarray, size: int = 1024) -> torch.Tensor:
    img = rgb_hw3.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size))
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def run_tile_inference(
    input_tif: str,
    model: nn.Module,
    tile_size: int   = 1024,
    overlap: int     = 128,
    rgb_bands: tuple = (1, 2, 3),
    threshold: float = 0.5,
    device: str      = "cuda",
    verbose: bool    = True,
) -> dict:
    """
    Sliding-window tile inference on a full GeoTIF with linear-weight blending.

    Returns
    -------
    dict: mask, prob_map, transform, crs, profile, total_tiles
    """
    model.eval().to(device)

    with rasterio.open(input_tif) as src:
        H, W       = src.height, src.width
        transform  = src.transform
        crs        = src.crs
        profile    = src.profile.copy()

    step = tile_size - overlap
    rows = list(range(0, H - tile_size + 1, step))
    cols = list(range(0, W - tile_size + 1, step))
    if not rows or rows[-1] + tile_size < H: rows.append(max(H - tile_size, 0))
    if not cols or cols[-1] + tile_size < W: cols.append(max(W - tile_size, 0))

    accum  = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)

    ramp = np.ones(tile_size, np.float32)
    ramp[:overlap]  = np.linspace(0, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0, overlap)
    w2d = np.outer(ramp, ramp)

    total = len(rows) * len(cols)
    report_every = 10 if total < 1000 else 100
    if verbose:
        print(f"TIF inference: {H}x{W} px | {total} tiles "
              f"({len(rows)} rows x {len(cols)} cols)")

    with rasterio.open(input_tif) as src:
        for ti, rs in enumerate(rows):
            for ci, cs in enumerate(cols):
                idx  = ti * len(cols) + ci
                win  = rasterio.windows.Window(cs, rs, tile_size, tile_size)
                data = src.read(list(rgb_bands), window=win,
                                out_shape=(len(rgb_bands), tile_size, tile_size),
                                boundless=True, fill_value=0)
                tile_rgb = data.transpose(1, 2, 0)
                inp  = _preprocess_rgb(tile_rgb, tile_size).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(model(inp)[-1]).squeeze().cpu().numpy()

                re, ce = rs + tile_size, cs + tile_size
                accum[rs:re, cs:ce]  += prob * w2d
                weight[rs:re, cs:ce] += w2d

                if verbose and (idx + 1) % report_every == 0:
                    print(f"  {idx+1}/{total} tiles processed...")

    weight   = np.where(weight == 0, 1, weight)
    prob_map = accum / weight
    mask     = (prob_map > threshold).astype(np.uint8)

    if verbose:
        print(f"Inference done. Positive px: {mask.sum():,} "
              f"({100*mask.mean():.3f}%)")

    profile.update(count=1, dtype='uint8', nodata=0)
    return dict(mask=mask, prob_map=prob_map,
                transform=transform, crs=crs, profile=profile,
                total_tiles=total)


# =============================================================================
# 8b. Full-Island Inference (memory-efficient streaming)
# =============================================================================

def run_island_full_inference(
    input_tif: str,
    boundary_shp: str,
    model: nn.Module,
    output_tif: str,
    tile_size: int       = 1024,
    overlap: int         = 128,
    threshold: float     = 0.5,
    use_tta: bool        = True,
    filter_canopy: bool  = True,
    canopy_cover_ratio: float = 0.80,
    building_gpkg: str   = None,
    min_length_m: float  = 0.0,
    rgb_bands: tuple     = (1, 2, 3),
    device: str          = "cuda",
    verbose: bool        = True,
) -> dict:
    """
    Full-resolution sliding-window inference on a large GeoTIFF, bounded by
    a polygon (e.g. Singapore Island boundary). Writes results directly to
    a tiled GeoTIFF — does NOT hold the full mosaic in memory.

    Memory usage: ~3 GB peak (one tile + downsampled boundary mask).

    Parameters
    ----------
    input_tif         : path to source GeoTIFF (e.g. SG_google_map_03m_SVY21.tif)
    boundary_shp      : polygon defining inference region (e.g. Island_boarder.shp)
    output_tif        : path to write the binary linkway mask
    tile_size, overlap: sliding window geometry
    use_tta           : run brightness-lifted second pass and average probs
    filter_canopy     : remove predictions ≥80 % under tree canopy (per-tile)

    Returns
    -------
    dict: output_tif, total_tiles, valid_tiles, transform, crs
    """
    model.eval().to(device)
    pad = overlap // 2
    step = tile_size - overlap

    # ── 1. Open input + boundary ──────────────────────────────────────────
    src = rasterio.open(input_tif)
    H, W       = src.height, src.width
    transform  = src.transform
    crs        = src.crs
    pix        = transform.a   # m / pixel (positive)
    if verbose:
        print(f"Input TIF : {W:,} x {H:,} px @ {pix} m  CRS: {crs}", flush=True)

    gdf = gpd.read_file(boundary_shp).to_crs(crs)
    if verbose:
        print(f"Boundary  : {len(gdf)} geoms, bounds={tuple(round(v) for v in gdf.total_bounds)}", flush=True)

    # ── 2. Downsampled boundary mask for fast tile filtering ─────────────
    DOWN = 16   # downsample factor
    H_d, W_d = H // DOWN, W // DOWN
    transform_d = rasterio.transform.from_origin(
        transform.c, transform.f, pix * DOWN, pix * DOWN
    )
    bnd_d = rasterize(
        [(g, 1) for g in gdf.geometry],
        out_shape=(H_d, W_d), transform=transform_d, fill=0, dtype=np.uint8,
    )
    if verbose:
        print(f"Boundary mask (1/{DOWN}) covers {100*bnd_d.mean():.2f}% of input area", flush=True)

    # ── 2b. Building footprint mask (rasterised once at 1/DOWN resolution) ──
    # Spatial-index queries on 300K polygons are too slow; instead we pre-
    # rasterise the entire building layer once to a downsampled uint8 array
    # and resample per-tile via nearest-neighbour. ~80 MB RAM, microseconds
    # per tile lookup.
    bld_mask_d = None
    if building_gpkg and os.path.exists(building_gpkg):
        if verbose:
            print(f"Building  : loading {building_gpkg}...", flush=True)
        bld_gdf = gpd.read_file(building_gpkg).to_crs(crs)
        if verbose:
            print(f"Building  : {len(bld_gdf):,} polygons — "
                  f"rasterising to 1/{DOWN} mask...", flush=True)
        bld_mask_d = rasterize(
            [(g, 1) for g in bld_gdf.geometry],
            out_shape=(H_d, W_d), transform=transform_d, fill=0, dtype=np.uint8,
        )
        # Free the GeoDataFrame — we only need the raster from now on
        del bld_gdf
        if verbose:
            print(f"Building  : mask covers {100*bld_mask_d.mean():.2f}% of input "
                  f"(shape={bld_mask_d.shape})", flush=True)

    # Min-length filter in pixels
    min_length_px = max(0, int(round(min_length_m / pix)))
    if verbose and min_length_px > 0:
        print(f"Length filter: blobs with major-axis < {min_length_m} m "
              f"({min_length_px} px) will be removed")

    # ── 3. Sliding window grid (tiles fully covering the input) ──────────
    rows = list(range(0, H - tile_size + 1, step))
    cols = list(range(0, W - tile_size + 1, step))
    if not rows or rows[-1] + tile_size < H: rows.append(max(H - tile_size, 0))
    if not cols or cols[-1] + tile_size < W: cols.append(max(W - tile_size, 0))

    # Filter to tiles whose bbox intersects the boundary
    valid = []
    for r in rows:
        for c in cols:
            r_d0, c_d0 = r // DOWN, c // DOWN
            r_d1 = (r + tile_size + DOWN - 1) // DOWN
            c_d1 = (c + tile_size + DOWN - 1) // DOWN
            if bnd_d[r_d0:r_d1, c_d0:c_d1].sum() > 0:
                valid.append((r, c))

    n_total = len(rows) * len(cols)
    n_valid = len(valid)
    if verbose:
        print(f"Total tiles : {n_total:,}  ({len(rows)} rows x {len(cols)} cols)", flush=True)
        print(f"Valid tiles : {n_valid:,}  ({100*n_valid/n_total:.1f}% — rest is water)", flush=True)

    # ── 4. Open output GeoTIFF (tiled, LZW-compressed, single uint8 band) ──
    out_profile = {
        'driver'   : 'GTiff',
        'dtype'    : 'uint8',
        'width'    : W,
        'height'   : H,
        'count'    : 1,
        'crs'      : crs,
        'transform': transform,
        'compress' : 'lzw',
        'tiled'    : True,
        'blockxsize': 512,
        'blockysize': 512,
        'nodata'   : 0,
        'BIGTIFF'  : 'YES',     # > 4 GB allowed
    }
    if verbose:
        print(f"Output TIF: {output_tif}")

    # ── 5. Inference loop ─────────────────────────────────────────────────
    report_every = max(1, n_valid // 100)
    pos_total    = 0
    t0           = time.time()

    with rasterio.open(output_tif, 'w', **out_profile) as dst:
        for idx, (r, c) in enumerate(valid):
            # Read input tile
            win  = rasterio.windows.Window(c, r, tile_size, tile_size)
            data = src.read(list(rgb_bands), window=win,
                            out_shape=(len(rgb_bands), tile_size, tile_size),
                            boundless=True, fill_value=0)
            tile_rgb = data.transpose(1, 2, 0).astype(np.uint8)

            # Skip if entirely black (no imagery)
            if tile_rgb.max() < 5:
                continue

            # Predict
            inp = _preprocess_rgb(tile_rgb, tile_size).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(inp)[-1]).squeeze().cpu().numpy()

                if use_tta:
                    img_b = np.clip(tile_rgb.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
                    inp_b = _preprocess_rgb(img_b, tile_size).to(device)
                    prob_b = torch.sigmoid(model(inp_b)[-1]).squeeze().cpu().numpy()
                    prob   = (prob + prob_b) / 2.0

            pred = (prob > threshold).astype(np.uint8)

            # Per-tile boundary clip (rasterize boundary at tile resolution)
            tile_tf = rasterio.windows.transform(win, transform)
            tile_bnd = rasterize(
                [(g, 1) for g in gdf.geometry],
                out_shape=(tile_size, tile_size), transform=tile_tf,
                fill=0, dtype=np.uint8,
            )
            pred = pred * tile_bnd

            # ── Building exclusion (downsampled mask + nearest upsample) ──
            if bld_mask_d is not None and pred.any():
                r_d0, c_d0 = r // DOWN, c // DOWN
                r_d1 = (r + tile_size + DOWN - 1) // DOWN
                c_d1 = (c + tile_size + DOWN - 1) // DOWN
                bld_tile_d = bld_mask_d[r_d0:r_d1, c_d0:c_d1]
                if bld_tile_d.sum() > 0:
                    # Resample to full tile resolution with nearest neighbour
                    bld_tile = cv2.resize(
                        bld_tile_d, (tile_size, tile_size),
                        interpolation=cv2.INTER_NEAREST
                    )
                    pred = pred * (1 - bld_tile)

            # ── Canopy filter (connected-component level) ─────────────────
            if filter_canopy and pred.any():
                canopy = compute_canopy_mask_np(tile_rgb)
                n_lbl, lbl_map = cv2.connectedComponents(pred)
                for lbl_id in range(1, n_lbl):
                    blob = (lbl_map == lbl_id)
                    if blob.sum() == 0:
                        continue
                    if canopy[blob].sum() / blob.sum() >= canopy_cover_ratio:
                        pred[blob] = 0

            # ── Min-length filter (5 m by default) ────────────────────────
            # Remove blobs whose longest dimension (rotated bbox) is too small.
            if min_length_px > 0 and pred.any():
                n_lbl2, lbl_map2 = cv2.connectedComponents(pred)
                for lbl_id in range(1, n_lbl2):
                    blob = (lbl_map2 == lbl_id).astype(np.uint8)
                    if blob.sum() == 0:
                        continue
                    contours, _ = cv2.findContours(
                        blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        continue
                    # minAreaRect on the largest contour
                    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
                    longest = max(rect[1])
                    if longest < min_length_px:
                        pred[blob == 1] = 0

            # Write central crop only — avoids edge artefacts where
            # tiles overlap. Outermost edges are written full.
            wr_top = 0           if r == rows[0] else pad
            wr_bot = tile_size   if r == rows[-1] else (tile_size - pad)
            wr_lft = 0           if c == cols[0] else pad
            wr_rht = tile_size   if c == cols[-1] else (tile_size - pad)

            crop = pred[wr_top:wr_bot, wr_lft:wr_rht]
            ow = wr_rht - wr_lft
            oh = wr_bot - wr_top
            # Clip to total raster bounds
            r_off = r + wr_top
            c_off = c + wr_lft
            ow = min(ow, W - c_off)
            oh = min(oh, H - r_off)
            if ow > 0 and oh > 0:
                dst.write(
                    crop[:oh, :ow], 1,
                    window=rasterio.windows.Window(c_off, r_off, ow, oh)
                )
                pos_total += int(crop[:oh, :ow].sum())

            if verbose and (idx + 1) % report_every == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (n_valid - idx - 1)
                print(f"  {idx+1:,}/{n_valid:,}  "
                      f"({100*(idx+1)/n_valid:.1f}%)  "
                      f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
                      flush=True)

    src.close()
    total_min = (time.time() - t0) / 60
    if verbose:
        print(f"\nInference done in {total_min:.1f} min")
        print(f"  Total positive px: {pos_total:,}")

    return dict(
        output_tif  = output_tif,
        total_tiles = n_total,
        valid_tiles = n_valid,
        transform   = transform,
        crs         = crs,
        positive_px = pos_total,
    )


# =============================================================================
# 8c. Full-Island Inference — PARALLEL (pipeline + GPU batch)
# =============================================================================

def run_island_full_inference_parallel(
    input_tif: str,
    boundary_shp: str,
    model: nn.Module,
    output_tif: str,
    tile_size: int        = 1024,
    overlap: int          = 128,
    threshold: float      = 0.5,
    use_tta: bool         = True,
    filter_canopy: bool   = True,
    canopy_cover_ratio: float = 0.80,
    building_gpkg: str    = None,
    min_length_m: float   = 0.0,
    min_aspect_ratio: float = 0.0,    # ≥3.0 keeps only elongated (linkway-like)
    rgb_bands: tuple      = (1, 2, 3),
    device: str           = "cuda",
    batch_size: int       = 8,
    use_amp_inference: bool = True,
    verbose: bool         = True,
) -> dict:
    """
    Parallel pipelined version of run_island_full_inference():
      Reader thread  : reads tiles + computes per-tile boundary/building/canopy masks
      GPU worker     : batches tiles, runs model (+ TTA), with FP16 autocast
      Post-proc thread: applies threshold + boundary + building + canopy + length filters
      Writer thread  : writes predictions to tiled GeoTIFF

    GPU batch size + AMP autocast give ~3-4x speedup vs the sequential version
    (GPU utilisation 4% -> 70-90%), while the 4-stage pipeline overlaps I/O
    with compute. TTA is preserved.
    """
    import threading
    from queue import Queue

    model.eval().to(device)
    pad  = overlap // 2
    step = tile_size - overlap
    min_length_px = max(0, int(round(min_length_m / 0.3)))   # 0.3 m placeholder

    # ── 1. Open input + boundary ──────────────────────────────────────────
    with rasterio.open(input_tif) as src_meta:
        H, W      = src_meta.height, src_meta.width
        transform = src_meta.transform
        crs       = src_meta.crs
    pix = transform.a
    min_length_px = max(0, int(round(min_length_m / pix)))
    if verbose:
        print(f"Input TIF : {W:,} x {H:,} px @ {pix} m  CRS: {crs}", flush=True)

    gdf = gpd.read_file(boundary_shp).to_crs(crs)
    if verbose:
        print(f"Boundary  : {len(gdf)} geoms", flush=True)

    # ── 2. Downsampled boundary mask ─────────────────────────────────────
    DOWN = 16
    H_d, W_d = H // DOWN, W // DOWN
    transform_d = rasterio.transform.from_origin(
        transform.c, transform.f, pix * DOWN, pix * DOWN
    )
    bnd_d = rasterize(
        [(g, 1) for g in gdf.geometry],
        out_shape=(H_d, W_d), transform=transform_d, fill=0, dtype=np.uint8,
    )
    del gdf
    if verbose:
        print(f"Boundary mask (1/{DOWN}) covers {100*bnd_d.mean():.2f}%", flush=True)

    # ── 2b. Building mask (rasterised once, downsampled) ─────────────────
    bld_mask_d = None
    if building_gpkg and os.path.exists(building_gpkg):
        if verbose:
            print(f"Building  : loading {building_gpkg}...", flush=True)
        bld_gdf = gpd.read_file(building_gpkg).to_crs(crs)
        if verbose:
            print(f"Building  : {len(bld_gdf):,} polygons -> 1/{DOWN} mask...", flush=True)
        bld_mask_d = rasterize(
            [(g, 1) for g in bld_gdf.geometry],
            out_shape=(H_d, W_d), transform=transform_d, fill=0, dtype=np.uint8,
        )
        del bld_gdf
        if verbose:
            print(f"Building  : mask covers {100*bld_mask_d.mean():.2f}%", flush=True)

    if verbose and min_length_px > 0:
        print(f"Length filter: < {min_length_m} m ({min_length_px} px) removed",
              flush=True)

    # ── 3. Sliding window grid + boundary filter ─────────────────────────
    rows = list(range(0, H - tile_size + 1, step))
    cols = list(range(0, W - tile_size + 1, step))
    if not rows or rows[-1] + tile_size < H: rows.append(max(H - tile_size, 0))
    if not cols or cols[-1] + tile_size < W: cols.append(max(W - tile_size, 0))

    valid = []
    for r in rows:
        for c in cols:
            r_d0, c_d0 = r // DOWN, c // DOWN
            r_d1 = (r + tile_size + DOWN - 1) // DOWN
            c_d1 = (c + tile_size + DOWN - 1) // DOWN
            if bnd_d[r_d0:r_d1, c_d0:c_d1].sum() > 0:
                valid.append((r, c))

    n_total = len(rows) * len(cols)
    n_valid = len(valid)
    if verbose:
        print(f"Total tiles : {n_total:,}  ({len(rows)} rows x {len(cols)} cols)", flush=True)
        print(f"Valid tiles : {n_valid:,}  ({100*n_valid/n_total:.1f}%)", flush=True)
        print(f"Pipeline    : reader -> GPU(batch={batch_size}, "
              f"TTA={use_tta}, AMP={use_amp_inference}) -> post -> writer", flush=True)

    # ── 4. Output GeoTIFF profile ─────────────────────────────────────────
    out_profile = {
        'driver': 'GTiff', 'dtype': 'uint8',
        'width': W, 'height': H, 'count': 1,
        'crs': crs, 'transform': transform,
        'compress': 'lzw', 'tiled': True,
        'blockxsize': 512, 'blockysize': 512,
        'nodata': 0, 'BIGTIFF': 'YES',
    }

    # ── 5. Pipeline: queues + worker threads ─────────────────────────────
    Q_TILES = Queue(maxsize=batch_size * 4)   # raw + masks → GPU
    Q_PREDS = Queue(maxsize=batch_size * 4)   # GPU output → post
    Q_WRITE = Queue(maxsize=batch_size * 4)   # post → writer

    progress = {'count': 0, 'pos': 0, 'lock': threading.Lock()}
    report_every = max(1, n_valid // 100)
    t0 = time.time()

    # ── Reader thread ────────────────────────────────────────────────────
    def reader_fn():
        try:
            with rasterio.open(input_tif) as rdr:
                for r, c in valid:
                    win = rasterio.windows.Window(c, r, tile_size, tile_size)
                    data = rdr.read(list(rgb_bands), window=win,
                                    out_shape=(len(rgb_bands), tile_size, tile_size),
                                    boundless=True, fill_value=0)
                    tile_rgb = data.transpose(1, 2, 0).astype(np.uint8)
                    if tile_rgb.max() < 5:
                        # Skip black tiles — still need progress update
                        with progress['lock']:
                            progress['count'] += 1
                        continue

                    # Per-tile masks from downsampled rasters (fast slice + resize)
                    r_d0, c_d0 = r // DOWN, c // DOWN
                    r_d1 = (r + tile_size + DOWN - 1) // DOWN
                    c_d1 = (c + tile_size + DOWN - 1) // DOWN
                    bnd_d_tile = bnd_d[r_d0:r_d1, c_d0:c_d1]
                    bnd_tile = cv2.resize(bnd_d_tile, (tile_size, tile_size),
                                          interpolation=cv2.INTER_NEAREST)

                    bld_tile = None
                    if bld_mask_d is not None:
                        bld_d_tile = bld_mask_d[r_d0:r_d1, c_d0:c_d1]
                        if bld_d_tile.sum() > 0:
                            bld_tile = cv2.resize(bld_d_tile, (tile_size, tile_size),
                                                  interpolation=cv2.INTER_NEAREST)

                    canopy_tile = (compute_canopy_mask_np(tile_rgb)
                                   if filter_canopy else None)

                    Q_TILES.put({
                        'r': r, 'c': c, 'tile_rgb': tile_rgb,
                        'bnd_tile': bnd_tile, 'bld_tile': bld_tile,
                        'canopy_tile': canopy_tile,
                    })
        finally:
            Q_TILES.put(None)   # sentinel

    # ── GPU worker thread ────────────────────────────────────────────────
    def gpu_worker_fn():
        try:
            buf = []
            def flush(buf):
                if not buf:
                    return
                B = len(buf)
                tiles = np.stack([
                    _preprocess_rgb(b['tile_rgb'], tile_size).numpy()[0]
                    for b in buf
                ])
                inp = torch.from_numpy(tiles).to(device, non_blocking=True)
                ctx = (torch.amp.autocast('cuda')
                       if (use_amp_inference and device == 'cuda')
                       else torch.cuda.amp.autocast(enabled=False))
                with torch.no_grad():
                    with ctx:
                        out = torch.sigmoid(model(inp)[-1])
                    probs_orig = out.float().squeeze(1).cpu().numpy()
                    if use_tta:
                        bright_tiles = np.stack([
                            _preprocess_rgb(
                                np.clip(b['tile_rgb'].astype(np.float32) * 1.25, 0, 255).astype(np.uint8),
                                tile_size
                            ).numpy()[0]
                            for b in buf
                        ])
                        inp_b = torch.from_numpy(bright_tiles).to(device, non_blocking=True)
                        with ctx:
                            out_b = torch.sigmoid(model(inp_b)[-1])
                        probs_bright = out_b.float().squeeze(1).cpu().numpy()
                        probs = (probs_orig + probs_bright) / 2.0
                    else:
                        probs = probs_orig
                for i, b in enumerate(buf):
                    b['prob'] = probs[i]
                    Q_PREDS.put(b)

            while True:
                item = Q_TILES.get()
                if item is None:
                    break
                buf.append(item)
                if len(buf) >= batch_size:
                    flush(buf); buf = []
            flush(buf)
        finally:
            Q_PREDS.put(None)

    # ── Post-process thread (bbox-localised, single pass) ────────────────
    # Use connectedComponentsWithStats for O(N_blobs) bbox metadata, then do
    # all filters within each blob's bbox window — avoids full-image scans
    # that made dense urban tiles >10x slower than rural ones.
    min_length_sq = (min_length_px * min_length_px) if min_length_px > 0 else 0

    def post_worker_fn():
        try:
            while True:
                item = Q_PREDS.get()
                if item is None:
                    break
                pred = (item['prob'] > threshold).astype(np.uint8)
                pred = pred * item['bnd_tile']
                if item['bld_tile'] is not None:
                    pred = pred * (1 - item['bld_tile'])

                if pred.any() and (filter_canopy or min_length_px > 0 or min_aspect_ratio > 0):
                    canopy = item['canopy_tile'] if filter_canopy else None

                    # One CC pass with stats (x, y, w, h, area) per blob
                    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(
                        pred, connectivity=8
                    )

                    need_rect = (min_length_px > 0) or (min_aspect_ratio > 0)

                    for lbl_id in range(1, n_lbl):
                        x, y, w, h, area = stats[lbl_id]
                        if area == 0:
                            continue
                        # Bbox window views (no full-image copy)
                        sub_lbl  = lbl_map[y:y+h, x:x+w]
                        local    = (sub_lbl == lbl_id)
                        pred_win = pred[y:y+h, x:x+w]

                        # ── Fast length filter via bbox-diagonal ──
                        # bbox_diag^2 = w^2 + h^2 upper-bounds the rotated
                        # bbox's longest side; if even that's too small,
                        # drop without an expensive minAreaRect call.
                        if min_length_sq > 0 and (w*w + h*h) < min_length_sq:
                            pred_win[local] = 0
                            continue

                        # ── Canopy filter (bbox-local) ──
                        if filter_canopy and canopy is not None:
                            sub_canopy = canopy[y:y+h, x:x+w]
                            ov = int(((sub_canopy != 0) & local).sum())
                            if ov / area >= canopy_cover_ratio:
                                pred_win[local] = 0
                                continue

                        # ── Length & aspect filter (single minAreaRect) ──
                        if need_rect:
                            local_u8 = local.astype(np.uint8)
                            contours, _ = cv2.findContours(
                                local_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            if contours:
                                rect = cv2.minAreaRect(
                                    max(contours, key=cv2.contourArea)
                                )
                                rw, rh   = rect[1]
                                longer   = max(rw, rh)
                                shorter  = max(min(rw, rh), 1.0)
                                aspect   = longer / shorter
                                if min_length_px > 0 and longer < min_length_px:
                                    pred_win[local] = 0
                                elif min_aspect_ratio > 0 and aspect < min_aspect_ratio:
                                    pred_win[local] = 0

                item['pred'] = pred
                Q_WRITE.put(item)
        finally:
            Q_WRITE.put(None)

    # ── Writer thread ────────────────────────────────────────────────────
    def writer_fn():
        with rasterio.open(output_tif, 'w', **out_profile) as dst:
            while True:
                item = Q_WRITE.get()
                if item is None:
                    break
                r, c, pred = item['r'], item['c'], item['pred']

                wr_top = 0           if r == rows[0]  else pad
                wr_bot = tile_size   if r == rows[-1] else (tile_size - pad)
                wr_lft = 0           if c == cols[0]  else pad
                wr_rht = tile_size   if c == cols[-1] else (tile_size - pad)

                crop = pred[wr_top:wr_bot, wr_lft:wr_rht]
                ow = wr_rht - wr_lft
                oh = wr_bot - wr_top
                r_off = r + wr_top
                c_off = c + wr_lft
                ow = min(ow, W - c_off)
                oh = min(oh, H - r_off)
                if ow > 0 and oh > 0:
                    dst.write(crop[:oh, :ow], 1,
                              window=rasterio.windows.Window(c_off, r_off, ow, oh))
                    p = int(crop[:oh, :ow].sum())
                else:
                    p = 0

                with progress['lock']:
                    progress['count'] += 1
                    progress['pos']   += p
                    if verbose and progress['count'] % report_every == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / progress['count'] * (n_valid - progress['count'])
                        print(f"  {progress['count']:,}/{n_valid:,}  "
                              f"({100*progress['count']/n_valid:.1f}%)  "
                              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
                              flush=True)

    # Launch all threads
    threads = [
        threading.Thread(target=reader_fn,    name="Reader",  daemon=False),
        threading.Thread(target=gpu_worker_fn, name="GPU",     daemon=False),
        threading.Thread(target=post_worker_fn, name="Post",   daemon=False),
        threading.Thread(target=writer_fn,    name="Writer",  daemon=False),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    total_min = (time.time() - t0) / 60
    if verbose:
        print(f"\nInference done in {total_min:.1f} min", flush=True)
        print(f"  Total positive px: {progress['pos']:,}", flush=True)

    return dict(
        output_tif  = output_tif,
        total_tiles = n_total,
        valid_tiles = n_valid,
        transform   = transform,
        crs         = crs,
        positive_px = progress['pos'],
    )


# =============================================================================
# 9. Per-tile Prediction Save  (for grid-based mosaic stitching)
# =============================================================================

def save_tile_predictions(
    model: nn.Module,
    images_dir: str,
    output_dir: str,
    splits: tuple          = ('train', 'val'),
    img_size: int          = 1024,
    threshold: float       = 0.5,
    device: str            = 'cuda',
    filter_canopy: bool    = True,
    canopy_exg_thresh: float = 0.12,
    canopy_cover_ratio: float = 0.80,
    verbose: bool          = True,
) -> int:
    """
    Run inference on every labeled tile PNG and save predicted mask as PNG.
    Output location: output_dir/{split}/tile_xxx.png  (0 / 255)

    Parameters
    ----------
    filter_canopy       : if True, remove predictions that fall almost entirely
                          under dense tree canopy (ExG-based vegetation index).
    canopy_exg_thresh   : ExG threshold to classify a pixel as canopy (default 0.12)
    canopy_cover_ratio  : a predicted linkway blob is removed when its canopy
                          overlap fraction exceeds this value (default 0.80 = 80%)
    """
    model.eval().to(device)
    total_written = 0

    for split in splits:
        in_dir  = os.path.join(images_dir, split)
        out_dir = os.path.join(output_dir,  split)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(in_dir, '*.png')))
        n = len(files)
        report_every = 10 if n < 1000 else 100
        if verbose:
            print(f"\n[{split}] Predicting {n} tiles "
                  f"(canopy filter={'ON' if filter_canopy else 'OFF'})...")

        for idx, fpath in enumerate(files):
            stem     = Path(fpath).stem
            out_path = os.path.join(out_dir, stem + '.png')

            img_rgb = np.array(Image.open(fpath).convert('RGB'), dtype=np.uint8)
            img_01  = img_rgb.astype(np.float32) / 255.0

            # TTA: run original + brightness-lifted version, average probabilities.
            # The brightened pass recovers linkways that sit in deep shadow.
            with torch.no_grad():
                inp_orig  = _preprocess_rgb(img_rgb, img_size).to(device)
                prob_orig = torch.sigmoid(model(inp_orig)[-1]).squeeze()

                # Brightness-lifted copy (+25 % intensity, clamped to 1)
                img_bright = np.clip(img_01 * 1.25, 0.0, 1.0)
                inp_bright = _preprocess_rgb(
                    (img_bright * 255).astype(np.uint8), img_size
                ).to(device)
                prob_bright = torch.sigmoid(model(inp_bright)[-1]).squeeze()

                # Average — upweights confidence where shadow depresses it
                prob = ((prob_orig + prob_bright) / 2.0).cpu().numpy()

            pred = (prob > threshold).astype(np.uint8)

            # ── Canopy post-processing ──────────────────────────────────
            if filter_canopy and pred.any():
                canopy = compute_canopy_mask_np(
                    img_rgb, exg_thresh=canopy_exg_thresh
                )
                # Label connected components; remove those ≥80% inside canopy
                n_lbl, lbl_map = cv2.connectedComponents(pred)
                for lbl_id in range(1, n_lbl):
                    blob = (lbl_map == lbl_id)
                    if blob.sum() == 0:
                        continue
                    canopy_frac = (canopy[blob]).sum() / blob.sum()
                    if canopy_frac >= canopy_cover_ratio:
                        pred[blob] = 0

            pred_out = (pred * 255).astype(np.uint8)
            Image.fromarray(pred_out).save(out_path)
            total_written += 1

            if verbose and (idx + 1) % report_every == 0:
                print(f"  [{split}] {idx+1}/{n} done...")

    if verbose:
        print(f"\nTotal tile predictions saved: {total_written}")
    return total_written


# =============================================================================
# 10. Grid-based Mosaic Stitching
# =============================================================================

def stitch_tiles_to_mosaic(
    pred_dir: str,
    grid_csv: str,
    input_tif: str,
    boundary_shp: str  = None,
    split_filter: str  = None,
    verbose: bool      = True,
) -> dict:
    """
    Stitch predicted tile PNGs into a full-resolution mosaic using
    col_off / row_off metadata from the grid CSV.

    Parameters
    ----------
    pred_dir     : directory with predicted PNG tiles (organized as {split}/tile_*.png
                   OR flat directory)
    grid_csv     : path to tiles_meta.csv (must have: filename, col_off, row_off)
    input_tif    : reference raster for extent, transform, CRS
    boundary_shp : optional SHP/GPKG to clip final mosaic to study area
    split_filter : 'train', 'val', or None (use all rows)
    verbose      : bool

    Returns
    -------
    dict: mask, transform, crs, profile
    """
    df = pd.read_csv(grid_csv)
    if split_filter and 'split' in df.columns:
        df = df[df['split'] == split_filter].reset_index(drop=True)

    with rasterio.open(input_tif) as src:
        H, W      = src.height, src.width
        transform = src.transform
        crs       = src.crs
        profile   = src.profile.copy()

    mosaic = np.zeros((H, W), dtype=np.uint8)
    n_placed  = 0
    n_missing = 0
    n_total   = len(df)
    report_every = 10 if n_total < 1000 else 100

    if verbose:
        print(f"Stitching {n_total} tiles into {H}x{W} mosaic...")

    for _, row in df.iterrows():
        fname   = row['filename']
        col_off = int(row['col_off'])
        row_off = int(row['row_off'])
        tw      = int(row.get('width',  1024))
        th      = int(row.get('height', 1024))

        # Search pred_dir and pred_dir/{split}
        found = None
        candidates = [
            os.path.join(pred_dir, fname),
            os.path.join(pred_dir, row.get('split', ''), fname) if 'split' in row.index else None,
        ]
        for c in candidates:
            if c and os.path.exists(c):
                found = c
                break

        if found is None:
            n_missing += 1
            continue

        tile = np.array(Image.open(found).convert('L'))
        tile_bin = (tile > 128).astype(np.uint8)

        # Resize if needed
        if tile_bin.shape != (th, tw):
            tile_bin = cv2.resize(tile_bin, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Clamp to mosaic boundary
        r0, r1 = row_off, min(row_off + th, H)
        c0, c1 = col_off, min(col_off + tw, W)
        pr, pc  = r1 - r0, c1 - c0
        mosaic[r0:r1, c0:c1] = np.maximum(
            mosaic[r0:r1, c0:c1], tile_bin[:pr, :pc]
        )
        n_placed += 1

        if verbose and n_placed % report_every == 0:
            print(f"  {n_placed}/{n_total} tiles placed...")

    if verbose:
        print(f"Stitch done: {n_placed} placed, {n_missing} not found")
        print(f"  Mosaic positive px: {mosaic.sum():,}")

    # Clip to boundary polygon
    if boundary_shp and os.path.exists(boundary_shp):
        gdf = gpd.read_file(boundary_shp).to_crs(crs)
        boundary_raster = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8,
        )
        mosaic = (mosaic * boundary_raster).astype(np.uint8)
        if verbose:
            print(f"  After boundary clip: {mosaic.sum():,} positive px")

    profile.update(count=1, dtype='uint8', nodata=0)
    return dict(mask=mosaic, transform=transform, crs=crs, profile=profile)


def stitch_from_tile_metadata(
    pred_dir: str,
    grid_csv: str,
    boundary_shp: str,
    pixel_size: float = 0.3,
    verbose: bool     = True,
) -> dict:
    """
    Memory-efficient mosaic stitcher using spatial transform metadata.

    Unlike stitch_tiles_to_mosaic(), this does NOT require opening the full
    82 GB reference TIF. It computes the output raster extent from
    boundary_shp and positions each tile using transform_c / transform_f
    (SVY21 coordinates) stored in grid_csv.

    Parameters
    ----------
    pred_dir     : directory with predicted PNGs
                   - accepts flat layout (pred_dir/filename.png)
                   - OR split layout (pred_dir/{split}/filename.png)
    grid_csv     : tiles_meta.csv — must have columns:
                   filename, split, transform_c, transform_f, width, height
    boundary_shp : SHP/GPKG of study area (sets output raster extent)
    pixel_size   : output pixel size in CRS units (default 0.3 m for EPSG:3414)
    verbose      : bool

    Returns
    -------
    dict: mask (H×W uint8), transform (rasterio Affine), crs, profile
    """
    df = pd.read_csv(grid_csv)

    # ── Compute output raster extent from boundary ─────────────────────────
    crs = rasterio.crs.CRS.from_epsg(3414)
    gdf_bnd = gpd.read_file(boundary_shp).to_crs(crs)
    xmin, ymin, xmax, ymax = gdf_bnd.total_bounds

    W = int(np.ceil((xmax - xmin) / pixel_size))
    H = int(np.ceil((ymax - ymin) / pixel_size))
    transform = rasterio.transform.from_origin(xmin, ymax, pixel_size, pixel_size)

    if verbose:
        print(f"Output mosaic : {W:,} × {H:,} px  @ {pixel_size} m  (EPSG:3414)")
        print(f"Boundary bbox : ({xmin:.0f}, {ymin:.0f}) → ({xmax:.0f}, {ymax:.0f})")
        print(f"Tiles to place: {len(df)}")

    mosaic    = np.zeros((H, W), dtype=np.uint8)
    n_placed  = 0
    n_missing = 0
    report_every = 10 if len(df) < 1000 else 100

    for _, row in df.iterrows():
        fname = row['filename']
        stem  = Path(fname).stem

        # Tile top-left corner in SVY21 metres
        tile_left = float(row['transform_c'])
        tile_top  = float(row['transform_f'])
        tw = int(row.get('width',  1024))
        th = int(row.get('height', 1024))

        # Convert to output pixel offset
        c0 = int(round((tile_left - xmin) / pixel_size))
        r0 = int(round((ymax   - tile_top) / pixel_size))

        # Skip tiles entirely outside boundary
        if r0 >= H or c0 >= W or r0 + th <= 0 or c0 + tw <= 0:
            n_missing += 1
            continue

        # Locate predicted tile PNG
        split = row.get('split', '') if 'split' in row.index else ''
        candidates = [
            os.path.join(pred_dir, fname),
            os.path.join(pred_dir, split, fname),
            os.path.join(pred_dir, stem + '.png'),
            os.path.join(pred_dir, split, stem + '.png'),
        ]
        found = next((c for c in candidates if c and os.path.exists(c)), None)

        if found is None:
            n_missing += 1
            continue

        tile = np.array(Image.open(found).convert('L'))
        tile_bin = (tile > 128).astype(np.uint8)

        # Clip destination window to mosaic bounds
        pr0 = max(r0, 0);  pc0 = max(c0, 0)
        pr1 = min(r0 + th, H);  pc1 = min(c0 + tw, W)
        tr0 = pr0 - r0;  tc0 = pc0 - c0
        tr1 = tr0 + (pr1 - pr0)
        tc1 = tc0 + (pc1 - pc0)

        if pr1 <= pr0 or pc1 <= pc0:
            n_missing += 1
            continue

        mosaic[pr0:pr1, pc0:pc1] = np.maximum(
            mosaic[pr0:pr1, pc0:pc1],
            tile_bin[tr0:tr1, tc0:tc1]
        )
        n_placed += 1
        if verbose and n_placed % report_every == 0:
            print(f"  {n_placed}/{len(df)} tiles placed...")

    if verbose:
        print(f"Stitch done: {n_placed} placed, {n_missing} not found / OOB")
        print(f"  Positive px: {mosaic.sum():,}  ({100 * mosaic.mean():.3f}%)")

    # ── Clip to boundary polygon ───────────────────────────────────────────
    gdf_bnd2 = gpd.read_file(boundary_shp).to_crs(crs)
    bnd_raster = rasterize(
        [(geom, 1) for geom in gdf_bnd2.geometry],
        out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8,
    )
    mosaic = (mosaic * bnd_raster).astype(np.uint8)
    if verbose:
        print(f"  After boundary clip: {mosaic.sum():,} positive px")

    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': W,
        'height': H,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'nodata': 0,
    }
    return dict(mask=mosaic, transform=transform, crs=crs, profile=profile)


# =============================================================================
# 11. Connectivity Post-processing
# =============================================================================

def connectivity_postprocess(
    mask: np.ndarray,
    closing_radius: int = 5,
    min_area_px: int    = 50,
    verbose: bool       = True,
) -> np.ndarray:
    """
    Morphological closing (fills small gaps) + remove tiny components.

    Parameters
    ----------
    closing_radius : radius of circular structuring element for closing
    min_area_px    : minimum component size in pixels to keep
    """
    struct  = np.ones((2 * closing_radius + 1,) * 2, dtype=bool)
    closed  = binary_closing(mask.astype(bool), structure=struct).astype(np.uint8)
    labeled, n = scipy_label(closed)

    if verbose:
        print(f"Connectivity post-process: {n} components (before area filter)")

    counts = np.bincount(labeled.ravel())
    keep   = set(np.where(counts >= min_area_px)[0]) - {0}
    out    = np.isin(labeled, list(keep)).astype(np.uint8)

    if verbose:
        removed = n - len(keep)
        print(f"  Removed {removed} components < {min_area_px} px")
        print(f"  Kept {len(keep)} components, {out.sum():,} px total")

    return out


# =============================================================================
# 12. GIS Constraints
# =============================================================================

def apply_gis_constraints(
    mask: np.ndarray,
    transform,
    crs,
    building_path: str,
    footpath_path: str          = None,
    footpath_buffer_m: float    = 2.0,
    building_overlap_thr: float = 0.5,
    verbose: bool               = True,
) -> tuple:
    """
    Apply spatial constraints:
    (A) Remove connected components with > building_overlap_thr overlap
        with building footprints.
    (B) Optionally keep only components intersecting a footpath buffer.

    Returns
    -------
    (mask_gis, building_gdf, footpath_gdf or None)
    """
    H, W = mask.shape

    # ── Building filter ──────────────────────────────────────────────────────
    bld_gdf     = gpd.read_file(building_path).to_crs(crs)
    bld_raster  = rasterize(
        [(g, 1) for g in bld_gdf.geometry],
        out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8,
    )
    labeled, n  = scipy_label(mask)
    out_mask    = np.zeros_like(mask)
    removed_bld = 0

    for cid in range(1, n + 1):
        comp    = (labeled == cid)
        overlap = (comp & bld_raster.astype(bool)).sum() / (comp.sum() + 1e-6)
        if overlap > building_overlap_thr:
            removed_bld += 1
        else:
            out_mask[comp] = 1

    if verbose:
        print(f"Building filter: {removed_bld}/{n} components removed "
              f"(overlap > {building_overlap_thr:.0%})")

    # ── Footpath filter ──────────────────────────────────────────────────────
    fp_gdf = None
    if footpath_path and os.path.exists(footpath_path):
        fp_gdf  = gpd.read_file(footpath_path).to_crs(crs)
        fp_buf  = fp_gdf.copy()
        fp_buf['geometry'] = fp_gdf.geometry.buffer(footpath_buffer_m)
        fp_raster = rasterize(
            [(g, 1) for g in fp_buf.geometry],
            out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8,
        )
        labeled2, n2 = scipy_label(out_mask)
        filtered     = np.zeros_like(out_mask)
        removed_fp   = 0

        for cid in range(1, n2 + 1):
            comp = (labeled2 == cid)
            if (comp & fp_raster.astype(bool)).any():
                filtered[comp] = 1
            else:
                removed_fp += 1

        if verbose:
            print(f"Footpath filter: {removed_fp}/{n2} components removed "
                  f"(no intersection with {footpath_buffer_m}m buffer)")
        out_mask = filtered

    if verbose:
        print(f"GIS final: {out_mask.sum():,} positive px")

    return out_mask.astype(np.uint8), bld_gdf, fp_gdf


# =============================================================================
# 13. Save Outputs
# =============================================================================

def save_mask_tif(mask: np.ndarray, out_path: str, profile: dict) -> str:
    """Save uint8 binary mask as compressed GeoTIFF."""
    p = profile.copy()
    p.update(count=1, dtype='uint8', compress='lzw', nodata=0,
             photometric=None)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with rasterio.open(out_path, 'w', **p) as dst:
        dst.write(mask.astype(np.uint8), 1)
    print(f"Raster saved: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


def mask_to_polygons(mask: np.ndarray, transform, crs,
                     min_area_m2: float = 2.0,
                     pixel_size_m: float = 0.3) -> gpd.GeoDataFrame:
    """Vectorize raster mask to GeoDataFrame, filtering by minimum area."""
    rows = []
    for geom_dict, val in rio_shapes(mask.astype(np.uint8), transform=transform):
        if val != 1:
            continue
        s    = shapely_shape(geom_dict)
        area = s.area
        if area >= min_area_m2:
            rows.append({'geometry': s, 'area_m2': round(area, 2)})
    if not rows:
        return gpd.GeoDataFrame(columns=['geometry', 'area_m2'], crs=crs)
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf = gdf.sort_values('area_m2', ascending=False).reset_index(drop=True)
    return gdf


def save_vector(gdf: gpd.GeoDataFrame, output_dir: str,
                name: str = 'covered_linkway') -> tuple:
    """Save GeoDataFrame as GPKG + SHP."""
    os.makedirs(output_dir, exist_ok=True)
    gpkg = os.path.join(output_dir, f"{name}.gpkg")
    shp  = os.path.join(output_dir, f"{name}.shp")
    if len(gdf) > 0:
        gdf.to_file(gpkg, driver='GPKG')
        gdf.to_file(shp)
        print(f"Vector saved: {gpkg}  ({len(gdf):,} polygons)")
    else:
        print("[warn] Empty GeoDataFrame — skipping vector save.")
    return gpkg, shp


# =============================================================================
# 14. Checkpoint Utilities
# =============================================================================

def list_checkpoints(save_dir: str) -> list:
    """List .pth files in save_dir sorted by val_dice (descending)."""
    ckpts = sorted(glob.glob(os.path.join(save_dir, '*.pth')))
    if not ckpts:
        print(f"No checkpoints found in {save_dir}")
        return []
    print(f"Found {len(ckpts)} checkpoint(s) in {save_dir}:")
    for i, p in enumerate(ckpts):
        mb = os.path.getsize(p) / 1e6
        print(f"  [{i}] {os.path.basename(p)}  ({mb:.0f} MB)")
    return ckpts


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    device: str = 'cuda') -> nn.Module:
    """Load state dict into model and move to device."""
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    print(f"Loaded: {os.path.basename(ckpt_path)}")
    return model
