"""
GeoSAM + TopoLoRA — AUTONOMOUS (point-free) training. Lean-checkpoint variant.
Written to C:/ temp to bypass D:/ filesystem corruption on new files.
"""

from __future__ import annotations

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from PIL import Image
import cv2

from segment_anything import sam_model_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo layout: shared training module lives in this folder
import subprocess as _sp

def _gpu_temp():
    """Return GPU core temp (int C) or None."""
    try:
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=8).decode().strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None

def _thermal_throttle(target=78, hard=80, max_wait=120, verbose=True):
    """If GPU temp > target, sleep in 5 s steps until it drops back to target
    (or max_wait reached). Returns seconds slept."""
    slept = 0
    t = _gpu_temp()
    if t is None or t <= target:
        return 0
    if verbose:
        print(f"  [thermal] GPU {t}C > {target}C -> cooling...", flush=True)
    while slept < max_wait:
        time.sleep(5)
        slept += 5
        t = _gpu_temp()
        if t is None or t <= target:
            break
    if verbose:
        print(f"  [thermal] resumed at {t}C after {slept}s", flush=True)
    return slept

from train_geosam_topolora_linkway import (
    inject_lora_into_qkv, cldice_loss, ensure_text_embedding,
)


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def structure_loss(pred_logits, mask, pos_weight=30.0):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    pw = torch.tensor(pos_weight, device=pred_logits.device, dtype=pred_logits.dtype)
    wbce = F.binary_cross_entropy_with_logits(pred_logits, mask,
                                              reduction="none", pos_weight=pw)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-6)
    pred_s = torch.sigmoid(pred_logits)
    inter  = ((pred_s * mask) * weit).sum(dim=(2, 3))
    union  = ((pred_s + mask) * weit).sum(dim=(2, 3))
    wiou   = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def _shadow_aug(img_01, prob=0.4):
    if np.random.rand() > prob:
        return img_01
    H, W = img_01.shape[:2]
    alpha = np.zeros((H, W), dtype=np.float32)
    if np.random.rand() > 0.5:
        r0 = np.random.randint(0, H); rw = np.random.randint(H // 10, H // 3)
        alpha[r0: min(r0 + rw, H), :] = 1.0
    else:
        c0 = np.random.randint(0, W); cw = np.random.randint(W // 10, W // 3)
        alpha[:, c0: min(c0 + cw, W)] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), np.random.randint(15, 50))
    darkness = np.random.uniform(0.30, 0.55)
    return np.clip(img_01 * (1.0 - alpha[..., None] * darkness), 0, 1).astype(np.float32)


class LinkwayDataset(Dataset):
    def __init__(self, images_dir, masks_dir, split, img_size=1024):
        self.image_dir = os.path.join(images_dir, split)
        self.mask_dir  = os.path.join(masks_dir,  split)
        self.split    = split
        self.img_size = img_size
        names = sorted(f for f in os.listdir(self.image_dir) if f.endswith(".png"))
        self.samples = [n for n in names
                        if os.path.exists(os.path.join(self.mask_dir, n))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"),
                       dtype=np.float32) / 255.0
        msk = np.array(Image.open(os.path.join(self.mask_dir,  name)).convert("L"),
                       dtype=np.float32)
        msk = (msk > 128).astype(np.float32)

        sz = self.img_size
        if img.shape[0] != sz or img.shape[1] != sz:
            img = cv2.resize(img, (sz, sz))
            msk = cv2.resize(msk, (sz, sz), interpolation=cv2.INTER_NEAREST)

        if self.split == "train":
            if np.random.rand() > 0.5: img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if np.random.rand() > 0.5: img = img[::-1].copy();    msk = msk[::-1].copy()
            k = np.random.choice([0, 1, 2, 3])
            if k:
                img = np.rot90(img, k).copy()
                msk = np.rot90(msk, k).copy()
            if np.random.rand() > 0.5:
                s = np.random.uniform(0.80, 1.0)
                ch = int(sz * s); cw = int(sz * s)
                r0 = np.random.randint(0, sz - ch + 1)
                c0 = np.random.randint(0, sz - cw + 1)
                img = cv2.resize(img[r0:r0+ch, c0:c0+cw], (sz, sz))
                msk = cv2.resize(msk[r0:r0+ch, c0:c0+cw], (sz, sz),
                                 interpolation=cv2.INTER_NEAREST)
            if np.random.rand() > 0.5:
                img = np.clip(img + np.random.uniform(-0.15, 0.15), 0, 1)
            if np.random.rand() > 0.5:
                f = np.random.uniform(0.80, 1.20)
                m = img.mean(axis=(0, 1), keepdims=True)
                img = np.clip((img - m) * f + m, 0, 1)
            if np.random.rand() > 0.35:
                img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.3, 1.5))
            img = _shadow_aug(img, prob=0.4)

        img = (img - _MEAN) / _STD
        return (torch.from_numpy(img.transpose(2, 0, 1).copy()),
                torch.from_numpy(msk).unsqueeze(0))


def forward_autonomous(sam, projection, text_emb, img_t, device, use_text=True):
    B, _, H, W = img_t.shape
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_STD,  device=device).view(1, 3, 1, 1)
    x_01  = (img_t * std + mean).clamp(0, 1)
    x_255 = (x_01 * 255).to(dtype=torch.float32)
    img_size = int(getattr(sam.image_encoder, "img_size", 1024))
    if x_255.shape[-1] != img_size or x_255.shape[-2] != img_size:
        x_255 = F.interpolate(x_255, size=(img_size, img_size),
                              mode="bilinear", align_corners=False)
    x_sam = sam.preprocess(x_255)
    image_embedding = sam.image_encoder(x_sam)

    with torch.no_grad():
        sparse_empty, dense_empty = sam.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
    text_token = None
    if use_text and text_emb is not None:
        emb_256 = projection(text_emb.float())
        text_token = emb_256.view(1, 1, -1).to(sparse_empty.dtype)
    sparse_b = torch.cat([sparse_empty, text_token], dim=1) if text_token is not None \
               else sparse_empty

    outs = []
    for b in range(B):
        low_res, _ = sam.mask_decoder(
            image_embeddings = image_embedding[b:b+1],
            image_pe         = sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_b,
            dense_prompt_embeddings  = dense_empty,
            multimask_output = False,
        )
        outs.append(low_res)
    return F.interpolate(torch.cat(outs, dim=0), (H, W), mode="bilinear", align_corners=False)


@torch.no_grad()
def validate(sam, projection, text_emb, loader, device, threshold=0.5):
    sam.eval()
    n, dice_sum, iou_sum = 0, 0.0, 0.0
    for imgs, msks in loader:
        imgs = imgs.to(device); msks = msks.to(device)
        logits = forward_autonomous(sam, projection, text_emb, imgs, device)
        pred = (torch.sigmoid(logits) > threshold).float()
        for b in range(pred.shape[0]):
            p = pred[b].cpu().numpy().astype(np.uint8).squeeze()
            g = msks[b].cpu().numpy().astype(np.uint8).squeeze()
            tp = (p & g).sum()
            fp = (p & (1 - g)).sum()
            fn = ((1 - p) & g).sum()
            dice_sum += 2 * tp / max(2 * tp + fp + fn, 1)
            iou_sum  += tp / max(tp + fp + fn, 1)
            n += 1
    sam.train()
    return dice_sum / max(n, 1), iou_sum / max(n, 1), n


def _trainable_sam_state(sam):
    return {k: v for k, v in sam.state_dict().items()
            if "lora_A" in k or "lora_B" in k or k.startswith("mask_decoder")}


def save_lean(path, sam, projection, args, epoch, val_dice):
    payload = {
        "sam_trainable": _trainable_sam_state(sam),
        "projection":    projection.state_dict(),
        "lora_rank":  args.lora_rank, "lora_alpha": args.lora_alpha,
        "class_name": "Covered Linkway",
        "epoch": epoch, "val_dice": val_dice,
        "use_text": not args.no_text, "autonomous": True,
    }
    tmp = path + ".tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"  [WARN] save failed: {e}")
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass
        return False


def cleanup_top_k(save_dir, prefix, k):
    files = [f for f in os.listdir(save_dir)
             if f.startswith(prefix) and f.endswith(".pth")]
    def _dice(f):
        try: return float(f.replace(prefix, "").replace(".pth", ""))
        except Exception: return -1.0
    files.sort(key=_dice, reverse=True)
    for f in files[k:]:
        try: os.remove(os.path.join(save_dir, f))
        except Exception: pass


def load_resume_state(path, sam, projection, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if "sam_trainable" in state:
        sam_state = state["sam_trainable"]
    elif "sam" in state:
        full = state["sam"]
        sam_state = {k: v for k, v in full.items()
                     if "lora_A" in k or "lora_B" in k or k.startswith("mask_decoder")}
    else:
        raise ValueError(f"Unknown checkpoint format: {path}")
    sam.load_state_dict(sam_state, strict=False)
    projection.load_state_dict(state["projection"])
    print(f"[resume] loaded {len(sam_state)} sam-trainable keys")
    print(f"[resume] previous epoch={state.get('epoch','?')} "
          f"val_dice={state.get('val_dice','?')}")
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--pos_weight",   type=float, default=30.0)
    p.add_argument("--cldice_weight",type=float, default=0.3)
    p.add_argument("--lora_rank",    type=int,   default=16)
    p.add_argument("--lora_alpha",   type=float, default=32.0)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--sam_ckpt",
        default=r"D:\Claude\GeoSAM-TopoLoRA\checkpoints\sam_vit_h_4b8939.pth")
    p.add_argument("--save_dir",
        default=r"D:\Claude\GeoSAM-TopoLoRA\checkpoints\geosam_topolora_autonomous")
    p.add_argument("--images_dir",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\images")
    p.add_argument("--masks_dir",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks")
    p.add_argument("--text_cache",
        default=r"D:\Claude\GeoSAM-TopoLoRA\checkpoints\clip_linkway_emb.pth")
    p.add_argument("--val_every",    type=int,   default=1)
    p.add_argument("--no_text",      action="store_true")
    p.add_argument("--resume_from",  type=str,   default=None)
    p.add_argument("--keep_top_k",   type=int,   default=3)
    p.add_argument("--thermal_target",   type=int, default=78,
                   help="cool GPU back to this temp every 10 batches (0=off)")
    p.add_argument("--thermal_max_wait", type=int, default=120,
                   help="max seconds to wait per cooldown")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2024); np.random.seed(2024)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[init] loading SAM ViT-H base from {args.sam_ckpt}")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).to(device)
    for p_ in sam.parameters(): p_.requires_grad_(False)
    n_lora = inject_lora_into_qkv(sam.image_encoder, args.lora_rank, args.lora_alpha)
    sam.image_encoder.to(device)
    for p_ in sam.mask_decoder.parameters(): p_.requires_grad_(True)
    proj = nn.Linear(512, 256).to(device)

    n_trn = (sum(p_.numel() for p_ in sam.parameters() if p_.requires_grad)
             + sum(p_.numel() for p_ in proj.parameters()))
    n_tot = (sum(p_.numel() for p_ in sam.parameters())
             + sum(p_.numel() for p_ in proj.parameters()))
    print(f"[init] LoRA blocks={n_lora}  trainable={n_trn/1e6:.2f}M / "
          f"{n_tot/1e6:.1f}M ({100*n_trn/n_tot:.2f}%)")

    text_emb = None
    if not args.no_text:
        text_emb = ensure_text_embedding(args.text_cache,
                                         class_name="Covered Linkway",
                                         device=device)

    resume_dice = -1.0
    if args.resume_from and os.path.exists(args.resume_from):
        rs = load_resume_state(args.resume_from, sam, proj, device)
        try: resume_dice = float(rs.get("val_dice", -1.0))
        except Exception: resume_dice = -1.0

    tr_ds = LinkwayDataset(args.images_dir, args.masks_dir, "train")
    va_ds = LinkwayDataset(args.images_dir, args.masks_dir, "val")
    print(f"[init] train={len(tr_ds)}  val={len(va_ds)}")
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                           num_workers=0, pin_memory=True)
    n_batch = len(tr_loader)
    report_every = 10 if n_batch < 1000 else 100
    print(f"[init] {n_batch} batches/epoch  report every {report_every}")

    trainable = [p_ for p_ in sam.parameters() if p_.requires_grad] \
                + list(proj.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = GradScaler() if device == "cuda" else None
    print(f"[init] loss = structure(pw={args.pos_weight}) + "
          f"{args.cldice_weight}*clDice  |  lr={args.lr}  epochs={args.epochs}")

    best_dice = resume_dice
    best_path = None
    log: list = []
    log_path = os.path.join(args.save_dir, "train_log.json")
    if os.path.exists(log_path):
        try: log = json.load(open(log_path))
        except Exception: log = []
        print(f"[init] {len(log)} previous epoch entries in log")

    ckpt_prefix = "geosam_topolora_autonomous_lean_dice"

    for epoch in range(1, args.epochs + 1):
        sam.train()
        t0 = time.time()
        ep_loss = 0.0
        for bi, (imgs, msks) in enumerate(tr_loader, start=1):
            imgs = imgs.to(device); msks = msks.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    logits = forward_autonomous(sam, proj, text_emb, imgs, device,
                                                use_text=not args.no_text)
                    loss = structure_loss(logits, msks, pos_weight=args.pos_weight)
                    if args.cldice_weight > 0:
                        loss = loss + args.cldice_weight * cldice_loss(logits, msks)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                logits = forward_autonomous(sam, proj, text_emb, imgs, device,
                                            use_text=not args.no_text)
                loss = structure_loss(logits, msks, pos_weight=args.pos_weight)
                if args.cldice_weight > 0:
                    loss = loss + args.cldice_weight * cldice_loss(logits, msks)
                loss.backward(); optimizer.step()
            ep_loss += loss.item()
            if bi % report_every == 0:
                avg = ep_loss / bi
                print(f"  epoch {epoch:>2}/{args.epochs} batch {bi:>4}/{n_batch} "
                      f"loss={loss.item():.4f}  avg={avg:.4f}", flush=True)
            # Adaptive thermal throttle: every 10 batches, cool if GPU > 78C
            if args.thermal_target > 0 and bi % 10 == 0:
                _thermal_throttle(target=args.thermal_target,
                                  max_wait=args.thermal_max_wait)
        scheduler.step()
        avg_loss = ep_loss / max(n_batch, 1)
        elapsed = time.time() - t0

        val_dice = val_iou = -1.0
        if epoch % args.val_every == 0:
            val_dice, val_iou, _ = validate(sam, proj, text_emb, va_loader, device)

        saved = ""
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(args.save_dir,
                f"{ckpt_prefix}{val_dice:.4f}.pth")
            if save_lean(best_path, sam, proj, args, epoch=epoch, val_dice=val_dice):
                saved = "  [SAVED]"
                cleanup_top_k(args.save_dir, ckpt_prefix, args.keep_top_k)
            else:
                saved = "  [SAVE-FAILED]"

        line = (f"EPOCH {epoch:>2}/{args.epochs} | loss={avg_loss:.4f} | "
                f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f} | "
                f"{elapsed/60:.1f}min{saved}")
        print(line, flush=True)
        log.append({"epoch": epoch, "loss": avg_loss,
                    "val_dice": val_dice, "val_iou": val_iou,
                    "elapsed_sec": elapsed,
                    "saved_path": best_path if saved == "  [SAVED]" else None})
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    final_path = os.path.join(args.save_dir,
                              "geosam_topolora_autonomous_lean_final.pth")
    save_lean(final_path, sam, proj, args,
              epoch=args.epochs, val_dice=best_dice)
    print(f"\n[done] best ={best_path}\n        final={final_path}")


if __name__ == "__main__":
    main()
