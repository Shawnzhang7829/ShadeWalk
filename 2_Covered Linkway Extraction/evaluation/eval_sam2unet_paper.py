"""
Paper-protocol eval of the best SAM2-UNet+LoRA checkpoint on the SAME
303 v3 val tiles used for the GeoSAM/Focal-Tversky "(model)" rows.

Mirrors eval_ckpt_paper.py exactly (micro P/R/F1/IoU + mean clDice + mean
Betti-0, threshold 0.5, native-tile resolution, prompt-free, leakage-free)
but runs the SAM2-UNet forward instead of the GeoSAM forward.

Preprocessing matches cl_pipeline.LinkwayDataset(split='val'):
  RGB/255 -> resize to 1024 if needed -> (img-_MEAN)/_STD ; no augmentation.
Prediction matches cl_pipeline._validate: sigmoid(model(x)[-1]) > 0.5.
"""
import os, sys, glob, numpy as np, torch, cv2
from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import label as cc_label

SAM2_REPO = r"D:\Claude\Meta-SAM2\models\sam2-main"
PIPE_DIR  = r"D:\Claude\Meta-SAM2"
for p in (SAM2_REPO, PIPE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import cl_pipeline as clp
from cl_pipeline import _MEAN, _STD

SAM2_CKPT = os.path.join(SAM2_REPO, r"checkpoints\sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
BEST_CKPT = r"D:\Claude\Meta-SAM2\checkpoints\best_model_dice0.4856.pth"
IMG = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\images\val"
MSK = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val"
OUT = r"D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\FINAL_REPORT\sam2unet_val_metrics.txt"
THR, SZ = 0.5, 1024


def cldice(p, g, e=1e-7):
    if p.sum() == 0 and g.sum() == 0: return 1.0
    if p.sum() == 0 or g.sum() == 0:  return 0.0
    sp, sg = skeletonize(p > 0), skeletonize(g > 0)
    tp = (sp & (g > 0)).sum() / (sp.sum() + e)
    ts = (sg & (p > 0)).sum() / (sg.sum() + e)
    return 0.0 if tp + ts < e else float(2 * tp * ts / (tp + ts + e))


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    model = clp.SAM2UNetLoRA(sam2_repo=SAM2_REPO, sam2_checkpoint=SAM2_CKPT,
                             sam2_config=SAM2_CFG, lora_rank=16, lora_alpha=32.0)
    clp.load_checkpoint(model, BEST_CKPT, device=dev)
    model.eval()

    names = sorted(f for f in os.listdir(IMG) if f.endswith(".png"))
    names = [n for n in names if os.path.exists(os.path.join(MSK, n))]
    TP = FP = FN = 0
    clds, betti = [], []
    for i, n in enumerate(names):
        img = np.array(Image.open(os.path.join(IMG, n)).convert("RGB"), np.float32) / 255.0
        g = (np.array(Image.open(os.path.join(MSK, n)).convert("L")) > 128).astype(np.uint8)
        gh, gw = g.shape
        x = cv2.resize(img, (SZ, SZ)) if img.shape[:2] != (SZ, SZ) else img
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)
        xt = torch.from_numpy(x.copy())[None].to(dev)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            lo = model(xt)[-1]
        p = (torch.sigmoid(lo).squeeze().float().cpu().numpy() > THR).astype(np.uint8)
        if p.shape != (gh, gw):
            p = cv2.resize(p, (gw, gh), interpolation=cv2.INTER_NEAREST)
        TP += int((p & g).sum()); FP += int((p & (1 - g)).sum())
        FN += int(((1 - p) & g).sum())
        clds.append(cldice(p, g)); betti.append(abs(cc_label(p)[1] - cc_label(g)[1]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(names)}")

    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-9); IoU = TP / max(TP + FP + FN, 1)
    cD, B0 = float(np.mean(clds)), float(np.mean(betti))
    line = (f"SAM2-UNet+LoRA (best_model_dice0.4856) vs {len(names)} v3 val tiles "
            f"[paper protocol: model fwd, thr0.5, micro-avg, prompt-free]\n"
            f"P={P:.4f} R={R:.4f} F1={F1:.4f} IoU={IoU:.4f} "
            f"clDice={cD:.4f} Betti0={B0:.2f}\n")
    print("\n" + line)
    with open(OUT, "w") as f:
        f.write(line)
    print("written", OUT)


if __name__ == "__main__":
    main()
