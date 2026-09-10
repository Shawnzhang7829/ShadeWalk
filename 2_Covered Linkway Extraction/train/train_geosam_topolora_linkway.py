"""
GeoSAM + TopoLoRA training for Covered Linkway.

Combines:
  - GeoSAM (Rafi et al., 2024) multimodal prompt strategy
      * SAM ViT-H backbone (loaded from sam_vit_h_4b8939.pth)
      * CLIP "Covered Linkway" text embedding -> Linear(512->256) -> dense prompt
      * Random fg/bg point sampling from the GT mask -> sparse prompt
      * DiceCE segmentation loss on the predicted mask
  - TopoLoRA (Seglab):
      * LoRA (rank=16, alpha=32) injected into every attn.qkv layer of the
        SAM image encoder.  Base weights stay frozen; only LoRA A/B train.
      * clDice topology loss (soft skeletonization + skeleton Dice) added to
        the segmentation loss with weight 0.3 -- specifically tuned for
        linkway, which is wider than vessels but still topologically thin.

Only LoRA params + mask decoder + projection layer are trainable
(~5-7M params, vs 636M for the full ViT-H).  This is both faster and a
stronger regulariser for the ~1.6k training tiles we have.

The script is single-image per step (GeoSAM-style) but adds gradient
accumulation if you want a larger effective batch.
"""

from __future__ import annotations

import os, sys, time, json, math, argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
from tqdm import tqdm
import monai

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# =========================================================================
# 0. LoRA — low-rank adaptation of nn.Linear layers
# =========================================================================

class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B @ A @ x   (W frozen, A,B trainable)."""

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / max(r, 1)
        in_f, out_f  = linear.in_features, linear.out_features
        self.lora_A  = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.lora_B  = nn.Parameter(torch.zeros(out_f, r))
        # Freeze base weight
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x):                                       # noqa: D401
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora_into_qkv(image_encoder: nn.Module,
                         rank: int = 16, alpha: float = 32.0) -> int:
    """Replace every attn.qkv Linear with a LoRA-wrapped version. Returns count."""
    n = 0
    for name, module in image_encoder.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, r=rank, alpha=alpha)
            n += 1
    return n


# =========================================================================
# 1. clDice topology loss  (from Seglab cl_pipeline.py)
# =========================================================================

class SoftSkeletonize(nn.Module):
    """Differentiable skeletonization via iterated soft-open + residuals."""

    def __init__(self, num_iter: int = 40):
        super().__init__()
        self.num_iter = num_iter

    def _erode(self, x):
        p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
        return torch.min(p1, p2)

    def _dilate(self, x):
        return F.max_pool2d(x, (3, 3), 1, 1)

    def _open(self, x):
        return self._dilate(self._erode(x))

    def forward(self, x):
        skel = F.relu(x - self._open(x))
        for _ in range(self.num_iter - 1):
            x    = self._erode(x)
            skel = skel + F.relu(x - self._open(x))
        return skel


_SOFT_SKEL = SoftSkeletonize(num_iter=40)


def cldice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Centerline-Dice loss for binary masks."""
    pred = torch.sigmoid(pred_logits)
    sp   = _SOFT_SKEL(pred)
    st   = _SOFT_SKEL(target)
    tprec = (sp * target).sum() / (sp.sum() + 1e-6)
    tsens = (st * pred  ).sum() / (st.sum() + 1e-6)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-6)


# =========================================================================
# 2. CLIP text embedding helper (cached)
# =========================================================================

def ensure_text_embedding(cache_path: str,
                          class_name: str = "Covered Linkway",
                          device: str = "cuda") -> torch.Tensor:
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device).to(device)
    import clip
    model, _ = clip.load("ViT-B/32", device)
    sentence = (f"{class_name}: a long narrow elevated covered walkway "
                f"with a roof viewed from above in aerial imagery")
    tokens = clip.tokenize([sentence]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens).squeeze(0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(emb.detach().cpu(), cache_path)
    return emb.to(device)


# =========================================================================
# 3. Dataset
# =========================================================================

class LinkwayDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, split: str):
        self.image_dir = os.path.join(images_dir, split)
        self.mask_dir  = os.path.join(masks_dir,  split)
        names = sorted(f for f in os.listdir(self.image_dir) if f.endswith(".png"))
        self.samples = [n for n in names
                        if os.path.exists(os.path.join(self.mask_dir, n))]
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        msk = Image.open(os.path.join(self.mask_dir,  name)).convert("L")
        return self.t(img), self.t(msk), name


# =========================================================================
# 4. Random fg/bg point sampler (mirrors GeoSAM's utils.get_random_points)
# =========================================================================

def sample_points(mask01: np.ndarray, n_fg: int = 2000, n_bg: int = 1000):
    fg = np.argwhere(mask01 == 1)
    bg = np.argwhere(mask01 == 0)
    if len(fg) == 0 or len(bg) == 0:
        return np.empty((0, 2), int), np.empty((0, 2), int)
    fg = fg[:, [1, 0]]; bg = bg[:, [1, 0]]
    np.random.shuffle(fg); np.random.shuffle(bg)
    return fg[:n_fg], bg[:n_bg]


# =========================================================================
# 5. Forward pass for ONE image with GRADIENT flow through LoRA
# =========================================================================

def forward_one_image(sam, projection, text_emb,
                      image_u8: np.ndarray, mask_u8: np.ndarray,
                      device: str, use_text: bool = True):
    """
    Run SAM (with LoRA-adapted image encoder, frozen prompt encoder,
    trainable mask decoder) end-to-end with gradients flowing through
    LoRA A/B, mask decoder and projection layer.

    Returns full-resolution logits of shape [1, 1, H, W].
    """
    H, W = image_u8.shape[:2]
    # ---- 1. Image embedding (gradient flows through LoRA) ----
    img_t = torch.from_numpy(image_u8).permute(2, 0, 1).float()[None].to(device)
    img_t = sam.preprocess(img_t)                              # 1024x1024 padded
    image_embedding = sam.image_encoder(img_t)                 # [1, 256, 64, 64]

    # ---- 2. Sparse prompt: random fg/bg points from GT mask ----
    fg_xy, bg_xy = sample_points(mask_u8, n_fg=2000, n_bg=1000)
    sparse_pts = None
    if len(fg_xy) > 0 and len(bg_xy) > 0:
        rls = ResizeLongestSide(sam.image_encoder.img_size)
        all_pts = rls.apply_coords(np.concatenate([fg_xy, bg_xy], 0), (H, W))
        lbls    = np.array([1] * len(fg_xy) + [0] * len(bg_xy), dtype=np.float32)
        all_pts = torch.as_tensor(all_pts, dtype=torch.float, device=device)[None]
        lbls    = torch.as_tensor(lbls,    dtype=torch.float, device=device)[None]
        sparse_pts = (all_pts, lbls)

    # ---- 3. Frozen prompt encoder ONLY for points (we hand-craft the text
    # part because the stock prompt_encoder expects boxes=[B,N,4] corner
    # coords; GeoSAM monkey-patched it to inject a 256-d text embedding
    # directly, which we replicate here without touching the SAM source).
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=sparse_pts, boxes=None, masks=None,
        )

    # ---- 4. Inject the CLIP text embedding as an extra sparse token ----
    if use_text and text_emb is not None:
        emb_256 = projection(text_emb.float())                  # [256]
        text_token = emb_256.view(1, 1, -1).to(sparse_embeddings.dtype)
        # sparse_embeddings: [B, N, 256] — append the text token
        sparse_embeddings = torch.cat([sparse_embeddings, text_token], dim=1)

    # ---- 5. Mask decoder (trainable) ----
    low_res, _ = sam.mask_decoder(
        image_embeddings = image_embedding,
        image_pe         = sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings = sparse_embeddings,
        dense_prompt_embeddings  = dense_embeddings,
        multimask_output = False,
    )
    logits = F.interpolate(low_res, (H, W), mode="bilinear", align_corners=False)
    return logits


# =========================================================================
# 6. Validation
# =========================================================================

@torch.no_grad()
def validate(sam, projection, text_emb, loader, device,
             thresh: float = 0.0):
    sam.eval()
    n, dice_sum, iou_sum = 0, 0.0, 0.0
    for imgs, msks, _ in loader:
        img_u8 = (imgs[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        msk    = (msks[0].squeeze().numpy() > 0.5).astype(np.uint8)
        if msk.max() == 0:
            continue
        logits = forward_one_image(sam, projection, text_emb,
                                   img_u8, msk, device, use_text=True)
        pred = (logits.squeeze().cpu().numpy() > thresh).astype(np.uint8)
        tp = (pred & msk).sum()
        fp = (pred & (1 - msk)).sum()
        fn = ((1 - pred) & msk).sum()
        dice_sum += 2 * tp / max(2 * tp + fp + fn, 1)
        iou_sum  += tp / max(tp + fp + fn, 1)
        n += 1
    sam.train()
    return dice_sum / max(n, 1), iou_sum / max(n, 1), n


# =========================================================================
# 7. Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--sam_ckpt",
        default=r"D:\Claude\GeoSAM-TopoLoRA\checkpoints\sam_vit_h_4b8939.pth")
    parser.add_argument("--save_dir",
        default=r"D:\Claude\GeoSAM-TopoLoRA\checkpoints\geosam_topolora_runs")
    parser.add_argument("--images_dir",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\images")
    parser.add_argument("--masks_dir",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks")
    parser.add_argument("--text_cache",
        default=r"D:\Claude\GeoSAM-TopoLoRA\checkpoints\clip_linkway_emb.pth")
    parser.add_argument("--no_text",        action="store_true")
    parser.add_argument("--lora_rank",      type=int,   default=16)
    parser.add_argument("--lora_alpha",     type=float, default=32.0)
    parser.add_argument("--cldice_weight",  type=float, default=0.3)
    parser.add_argument("--val_every",      type=int,   default=1)
    parser.add_argument("--max_train_per_epoch", type=int, default=0,
        help="if >0, subsample this many training tiles per epoch")
    parser.add_argument("--grad_accum",     type=int,   default=1,
        help="gradient accumulation steps (effective batch size)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2023); np.random.seed(2023)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Build SAM ViT-H and freeze base weights ──
    print(f"[init] loading SAM ViT-H from {args.sam_ckpt}")
    assert os.path.exists(args.sam_ckpt), f"missing: {args.sam_ckpt}"
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).to(device)

    # Freeze everything first
    for p in sam.parameters():
        p.requires_grad_(False)

    # ── 2. LoRA injection into image encoder QKV ──
    n_lora = inject_lora_into_qkv(sam.image_encoder,
                                  rank=args.lora_rank,
                                  alpha=args.lora_alpha)
    sam.image_encoder.to(device)            # move newly added LoRA params
    print(f"[init] LoRA injected into {n_lora} QKV layers "
          f"(rank={args.lora_rank}, alpha={args.lora_alpha})")

    # ── 3. Unfreeze mask decoder (classic GeoSAM recipe) ──
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)

    # ── 4. Projection layer (CLIP 512 -> SAM box-embedding 256) ──
    projection = nn.Linear(512, 256).to(device)

    # ── 5. Param accounting ──
    n_total = sum(p.numel() for p in sam.parameters()) \
            + sum(p.numel() for p in projection.parameters())
    n_train = sum(p.numel() for p in sam.parameters() if p.requires_grad) \
            + sum(p.numel() for p in projection.parameters())
    print(f"[init] params total={n_total/1e6:.1f}M  "
          f"trainable={n_train/1e6:.2f}M  "
          f"({100*n_train/n_total:.2f}%)")

    # ── 6. CLIP text embedding ──
    text_emb = None
    if not args.no_text:
        text_emb = ensure_text_embedding(args.text_cache,
                                         class_name="Covered Linkway",
                                         device=device)
        print(f"[init] text embedding shape = {tuple(text_emb.shape)}")

    # ── 7. Data ──
    tr_ds = LinkwayDataset(args.images_dir, args.masks_dir, "train")
    va_ds = LinkwayDataset(args.images_dir, args.masks_dir, "val")
    print(f"[init] train={len(tr_ds)}  val={len(va_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=1, shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── 8. Optimiser + losses ──
    trainable = [p for p in sam.parameters() if p.requires_grad] \
              + list(projection.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler() if device == "cuda" else None

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, reduction="mean")
    cldice_w = args.cldice_weight
    print(f"[init] loss = DiceCE + {cldice_w} * clDice "
          f"(grad_accum={args.grad_accum})")

    # ── 9. Train ──
    n_per_epoch = len(tr_ds) if args.max_train_per_epoch <= 0 \
                  else min(args.max_train_per_epoch, len(tr_ds))
    report_every = 10 if n_per_epoch < 1000 else 100
    print(f"[init] report every {report_every} samples "
          f"(n_per_epoch={n_per_epoch})")

    best_dice = -1.0
    best_path = None
    log: list = []

    for epoch in range(1, args.epochs + 1):
        sam.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_seen = 0; n_skipped = 0
        optimizer.zero_grad()

        indices = list(range(len(tr_ds)))
        np.random.shuffle(indices)
        if args.max_train_per_epoch > 0:
            indices = indices[:args.max_train_per_epoch]

        for i, idx in enumerate(indices, start=1):
            img_t, msk_t, _ = tr_ds[idx]
            img_u8 = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            msk    = (msk_t.squeeze().numpy() > 0.5).astype(np.uint8)
            if msk.max() == 0:
                n_skipped += 1
                continue

            # AMP for memory; gradient flows through LoRA, decoder, projection
            if scaler is not None:
                with autocast():
                    logits = forward_one_image(
                        sam, projection, text_emb, img_u8, msk, device,
                        use_text=(not args.no_text),
                    )
                    target = torch.from_numpy(msk).float().to(device)[None, None]
                    loss = seg_loss(logits, target)
                    if cldice_w > 0:
                        loss = loss + cldice_w * cldice_loss(logits, target)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                if i % args.grad_accum == 0:
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad()
            else:
                logits = forward_one_image(
                    sam, projection, text_emb, img_u8, msk, device,
                    use_text=(not args.no_text),
                )
                target = torch.from_numpy(msk).float().to(device)[None, None]
                loss = seg_loss(logits, target)
                if cldice_w > 0:
                    loss = loss + cldice_w * cldice_loss(logits, target)
                loss = loss / args.grad_accum
                loss.backward()
                if i % args.grad_accum == 0:
                    optimizer.step(); optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            n_seen     += 1

            if i % report_every == 0:
                avg = epoch_loss / max(n_seen, 1)
                print(f"  epoch {epoch:>2}/{args.epochs} "
                      f"step {i:>4}/{len(indices)} "
                      f"loss={loss.item()*args.grad_accum:.4f}  "
                      f"avg={avg:.4f}", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / max(n_seen, 1)
        elapsed  = time.time() - t0

        # Validation
        val_dice = val_iou = -1.0
        if epoch % args.val_every == 0:
            val_dice, val_iou, _ = validate(
                sam, projection, text_emb, va_loader, device,
            )

        saved = ""
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(
                args.save_dir,
                f"geosam_topolora_linkway_best_dice{val_dice:.4f}.pth",
            )
            torch.save({
                "sam"       : sam.state_dict(),
                "projection": projection.state_dict(),
                "lora_rank" : args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "class_name": "Covered Linkway",
                "epoch"     : epoch,
                "val_dice"  : val_dice,
                "use_text"  : not args.no_text,
            }, best_path)
            saved = "  [SAVED]"

        line = (f"EPOCH {epoch:>2}/{args.epochs} | "
                f"loss={avg_loss:.4f} | "
                f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f} | "
                f"seen={n_seen}  skipped={n_skipped} | "
                f"{elapsed/60:.1f}min{saved}")
        print(line, flush=True)
        log.append({"epoch": epoch, "loss": avg_loss,
                    "val_dice": val_dice, "val_iou": val_iou,
                    "elapsed_sec": elapsed})
        with open(os.path.join(args.save_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    # Final snapshot
    final_path = os.path.join(args.save_dir,
                              "geosam_topolora_linkway_final.pth")
    torch.save({
        "sam"       : sam.state_dict(),
        "projection": projection.state_dict(),
        "lora_rank" : args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "class_name": "Covered Linkway",
        "epoch"     : args.epochs,
        "val_dice"  : best_dice,
        "use_text"  : not args.no_text,
    }, final_path)
    print(f"\n[done] best ={best_path}\n        final={final_path}")
    return best_path, final_path


if __name__ == "__main__":
    main()
