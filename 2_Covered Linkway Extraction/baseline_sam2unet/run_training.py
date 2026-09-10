"""
run_training.py  —  Covered Linkway one-click training script
Usage: conda activate sam2 && python D:\Claude\Meta-SAM2\run_training.py
"""
import os, sys, time

# ── Paths ──────────────────────────────────────────────────────────────────
SAM2_REPO    = r"D:\Claude\Meta-SAM2\models\sam2-main"
PIPELINE_DIR = r"D:\Claude\Meta-SAM2"

for _p in (SAM2_REPO, PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SAM2_CHECKPOINT = os.path.join(SAM2_REPO, r"checkpoints\sam2.1_hiera_large.pt")
# Verify checkpoint exists
assert os.path.exists(SAM2_CHECKPOINT), f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

DATA_BASE      = r"D:\Claude\Meta-SAM2\Covered Linkway"
IMAGES_DIR     = os.path.join(DATA_BASE, "images")
JSON_DIR       = os.path.join(DATA_BASE, "json")
MASKS_DIR      = os.path.join(DATA_BASE, "masks")
CHECKPOINT_DIR = r"D:\Claude\Meta-SAM2\checkpoints"

# ── Hyper-parameters ────────────────────────────────────────────────────────────────
TILE_SIZE          = 1024
BATCH_SIZE         = 4      # 2→4: more stable gradients (51.5GB VRAM, AMP FP16)
EPOCHS             = 50     # 20→50: room to reach IoU>0.8
LR                 = 1e-4
POS_WEIGHT         = 50.0
CLDICE_WEIGHT      = 0.3
LORA_RANK          = 16
LORA_ALPHA         = 32.0
DEVICE             = "cuda"
EARLY_STOP_PATIENCE = 15   # stop if no val_dice improvement for 15 epochs

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Covered Linkway Training")
print("=" * 60)

import torch
if DEVICE == "cuda" and not torch.cuda.is_available():
    print("[warn] CUDA not available, using CPU")
    DEVICE = "cpu"
else:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}  |  VRAM: {props.total_memory/1e9:.1f} GB")

# ── Step 1: Generate masks ────────────────────────────────────────────────
print("\n[Step 1/3] Generating masks from JSON annotations...")
import cl_pipeline as clp

stats = clp.generate_masks_from_json(
    images_dir=IMAGES_DIR,
    json_dir=JSON_DIR,
    masks_dir=MASKS_DIR,
    splits=('train', 'val'),
    verbose=True,
)
for split, s in stats.items():
    print(f"  {split}: {s['written']} masks, {s['empty']} empty")

# ── Step 2: Build model ───────────────────────────────────────────────────
print("\n[Step 2/3] Building SAM2-UNet+LoRA model...")
model = clp.SAM2UNetLoRA(
    sam2_repo       = SAM2_REPO,
    sam2_checkpoint = SAM2_CHECKPOINT,
    sam2_config     = SAM2_CONFIG,
    lora_rank       = LORA_RANK,
    lora_alpha      = LORA_ALPHA,
)
model = model.to(DEVICE)
print(clp.count_trainable_params(model))

# ── Step 3: Train ─────────────────────────────────────────────────────────
print("\n[Step 3/3] Loading data and starting training...")
train_loader, val_loader = clp.get_dataloaders(
    images_dir  = IMAGES_DIR,
    masks_dir   = MASKS_DIR,
    batch_size  = BATCH_SIZE,
    img_size    = TILE_SIZE,
    num_workers = 0,          # 0 on Windows: avoids multiprocessing spawn issues
)
print(f"Train: {len(train_loader.dataset)} samples "
      f"| Val: {len(val_loader.dataset)} samples")

t_start = time.time()
best_ckpt = clp.train(
    model                = model,
    train_loader         = train_loader,
    val_loader           = val_loader,
    epochs               = EPOCHS,
    lr                   = LR,
    weight_decay         = 5e-4,
    cldice_weight        = CLDICE_WEIGHT,
    pos_weight           = POS_WEIGHT,
    save_dir             = CHECKPOINT_DIR,
    device               = DEVICE,
    use_amp              = True,            # AMP ~2x faster on Ada GPU
    early_stop_patience  = EARLY_STOP_PATIENCE,
    verbose              = True,
)
total_time = time.time() - t_start

# ── Report ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training Complete")
print("=" * 60)
print(f"  Total time     : {total_time/3600:.2f} h  ({total_time:.0f} s)")
print(f"  Best checkpoint: {best_ckpt}")

import glob
ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
print(f"\nAll saved checkpoints ({len(ckpts)}):")
for p in ckpts:
    mb = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p)}  ({mb:.0f} MB)")
