"""
run_inference_mosaic.py — Full-Singapore Covered-Linkway Inference
==================================================================
Runs sliding-window inference on the entire SG_google_map_03m_SVY21.tif
(179K × 115K px @ 0.3 m), bounded by Island_boarder.shp. Writes a
single tiled, LZW-compressed GeoTIFF.

Usage:
  python run_inference_mosaic.py --wait-pid <TRAINING_PID>   # waits for training
  python run_inference_mosaic.py                              # runs immediately

Output:
  D:\\Claude\\Meta-SAM2\\outputs\\linkway_mask_full.tif    (BIGTIFF, EPSG:3414)
"""
import os, sys, glob, argparse, time

# ── Paths ─────────────────────────────────────────────────────────────────────
SAM2_REPO    = r"D:\Claude\Meta-SAM2\models\sam2-main"
PIPELINE_DIR = r"D:\Claude\Meta-SAM2"
for _p in (SAM2_REPO, PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SAM2_CHECKPOINT = os.path.join(SAM2_REPO, r"checkpoints\sam2.1_hiera_large.pt")
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

INPUT_TIF       = r"D:\Claude\Meta-SAM2\Covered Linkway\SG_google_map_03m_SVY21.tif"
BOUNDARY_SHP    = r"D:\Claude\Meta-SAM2\shp\Island_boarder\Island_boarder.shp"
BUILDING_GPKG   = r"C:\Users\City Syntax Lab\Desktop\SOLWEIG_GPU\data\1-SG data\Shp\SG_Building\SG_Building_SVY21.gpkg"
CHECKPOINT_DIR  = r"D:\Claude\Meta-SAM2\checkpoints"
OUTPUT_DIR      = r"D:\Claude\Meta-SAM2\outputs"
OUTPUT_TIF      = os.path.join(OUTPUT_DIR, "linkway_mask_full_v3.tif")

LORA_RANK       = 16
LORA_ALPHA      = 32.0
DEVICE          = "cuda"
THRESHOLD       = 0.7     # v3: 0.6 -> 0.7 (stricter, fewer false positives)
MIN_LENGTH_M    = 8.0     # v3: 5 m -> 8 m (remove more short fragments)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--wait-pid', type=int, default=None,
                    help='Wait for this PID to exit before starting inference')
parser.add_argument('--threshold', type=float, default=THRESHOLD,
                    help=f'Binarisation threshold (default {THRESHOLD})')
parser.add_argument('--no-tta', action='store_true',
                    help='Disable test-time augmentation (faster but lower recall in shadow)')
parser.add_argument('--no-canopy', action='store_true',
                    help='Disable canopy connected-component filter')
parser.add_argument('--no-building', action='store_true',
                    help='Disable building footprint exclusion')
parser.add_argument('--min-length', type=float, default=MIN_LENGTH_M,
                    help=f'Remove blobs shorter than this many metres (default {MIN_LENGTH_M})')
parser.add_argument('--ckpt', type=str, default=None,
                    help='Override best checkpoint with this path')
args = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Wait for training process ─────────────────────────────────────────────────
if args.wait_pid:
    import psutil
    print(f"[wait] Monitoring training PID {args.wait_pid}...")
    while psutil.pid_exists(args.wait_pid):
        time.sleep(60)
        if psutil.pid_exists(args.wait_pid):
            print(f"  PID {args.wait_pid} still running — "
                  f"{time.strftime('%H:%M:%S')}")
    print(f"[wait] PID {args.wait_pid} has exited. Starting inference.\n")

# ── Imports (after potential long wait) ───────────────────────────────────────
import torch
import cl_pipeline as clp

print("=" * 70)
print("  Covered Linkway — Full Singapore Inference")
print("=" * 70)

# ── Step 1: Find best checkpoint ──────────────────────────────────────────────
print("\n[1/3] Finding best checkpoint...")
if args.ckpt:
    best_ckpt = args.ckpt
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(best_ckpt)
else:
    ckpts = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "best_model_dice*.pth")),
        key=lambda p: float(os.path.basename(p).split('dice')[-1].replace('.pth', '')),
        reverse=True
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found in {CHECKPOINT_DIR}. "
            "Run run_training.py first or pass --ckpt."
        )
    best_ckpt = ckpts[0]

dice_val = float(os.path.basename(best_ckpt).split('dice')[-1].replace('.pth', ''))
print(f"  Using : {os.path.basename(best_ckpt)}  (val_dice={dice_val:.4f})")

# ── Step 2: Build model & load weights ───────────────────────────────────────
print("\n[2/3] Building SAM2-UNet+LoRA and loading weights...")
if DEVICE == 'cuda' and not torch.cuda.is_available():
    print("[warn] CUDA not available, falling back to CPU")
    DEVICE = 'cpu'
else:
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU : {p.name}  VRAM={p.total_memory/1e9:.1f} GB")

model = clp.SAM2UNetLoRA(
    sam2_repo       = SAM2_REPO,
    sam2_checkpoint = SAM2_CHECKPOINT,
    sam2_config     = SAM2_CONFIG,
    lora_rank       = LORA_RANK,
    lora_alpha      = LORA_ALPHA,
)
clp.load_checkpoint(model, best_ckpt, device=DEVICE)
model = model.to(DEVICE)
print(f"  Weights loaded.")

# ── Step 3: Full-island inference ─────────────────────────────────────────────
print("\n[3/3] Running sliding-window inference over full Singapore Island...")
print(f"  Input    : {INPUT_TIF}")
print(f"  Boundary : {BOUNDARY_SHP}")
print(f"  Building : {'(disabled)' if args.no_building else BUILDING_GPKG}")
print(f"  Output   : {OUTPUT_TIF}")
print(f"  Threshold      : {args.threshold}")
print(f"  Min length     : {args.min_length} m")
print(f"  Min aspect     : 3.0 (drop blobs that aren't elongated)")
print(f"  TTA            : {'OFF' if args.no_tta else 'ON (orig + 1.25x bright avg)'}")
print(f"  Canopy filter  : {'OFF' if args.no_canopy else 'ON (>=60% under canopy -> drop blob)'}\n")

t_start = time.time()
result = clp.run_island_full_inference_parallel(
    input_tif          = INPUT_TIF,
    boundary_shp       = BOUNDARY_SHP,
    model              = model,
    output_tif         = OUTPUT_TIF,
    tile_size          = 1024,
    overlap            = 128,
    threshold          = args.threshold,
    use_tta            = not args.no_tta,
    filter_canopy      = not args.no_canopy,
    canopy_cover_ratio = 0.60,      # v3 Phase-1: 0.80 -> 0.60 (stricter tree drop)
    building_gpkg      = (None if args.no_building else BUILDING_GPKG),
    min_length_m       = args.min_length,
    min_aspect_ratio   = 3.0,       # v3 Phase-1: keep elongated blobs only
    rgb_bands          = (1, 2, 3),
    device             = DEVICE,
    batch_size         = 8,         # GPU batch — uses ~5-6 GB VRAM with TTA
    use_amp_inference  = True,      # FP16 autocast at inference
    verbose            = True,
)
total_min = (time.time() - t_start) / 60

# ── Report ────────────────────────────────────────────────────────────────────
sz_gb = os.path.getsize(OUTPUT_TIF) / 1e9
print("\n" + "=" * 70)
print("  Done!")
print("=" * 70)
print(f"  Output TIF    : {OUTPUT_TIF}")
print(f"  File size     : {sz_gb:.2f} GB")
print(f"  Total tiles   : {result['total_tiles']:,}")
print(f"  Valid tiles   : {result['valid_tiles']:,}  (within Island_boarder)")
print(f"  Positive px   : {result['positive_px']:,}")
print(f"  Total time    : {total_min:.1f} min")
print(f"\nOpen {OUTPUT_TIF} in QGIS to verify (auto-georeferenced to SVY21).")
