"""
generate_masks.py — generate PNG masks from JSON annotations (stand-alone run)
"""
import os, sys

SAM2_REPO    = r"C:\Users\City Syntax Lab\Desktop\Covered Linkways\3-Segmentation code\sam2-main"
PIPELINE_DIR = r"D:\Claude\Meta-SAM2"
for _p in (SAM2_REPO, PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_BASE  = r"D:\Claude\Meta-SAM2\Covered Linkway"
IMAGES_DIR = os.path.join(DATA_BASE, "images")
JSON_DIR   = os.path.join(DATA_BASE, "json")
MASKS_DIR  = os.path.join(DATA_BASE, "masks")

import cl_pipeline as clp

print("Generating masks from JSON annotations...")
stats = clp.generate_masks_from_json(
    images_dir = IMAGES_DIR,
    json_dir   = JSON_DIR,
    masks_dir  = MASKS_DIR,
    splits     = ('train', 'val'),
    verbose    = True,
)

print("\n=== Summary ===")
for split, s in stats.items():
    print(f"  {split}: {s['written']}/{s['total']} masks written  "
          f"({s['empty']} empty, {s['skipped_existing']} already existed)")

# Quick sanity check: count generated files
import glob
for split in ('train', 'val'):
    n = len(glob.glob(os.path.join(MASKS_DIR, split, "*.png")))
    print(f"  masks/{split}/ : {n} PNG files on disk")
print("\nDone.")
