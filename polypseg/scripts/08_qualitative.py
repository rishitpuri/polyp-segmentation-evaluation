"""Render qualitative comparison panels.

Runs on the pod (needs checkpoints + GPU). For a set of automatically
selected fold-0 test images, produces a grid:

    input | ground truth | U-Net | U-Net++ | SegFormer-B0

with each prediction outlined against the reference contour and its Dice
printed, so failure modes are directly visible.

Cases are chosen from the fold-0 *test* split only, so no model has trained
on any image shown.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from polypseg.src.data import eval_transform, kvasir_records
from polypseg.src.metrics import region_metrics
from polypseg.src.models import build_model
from polypseg.src.train import pick_device

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["unet", "unetpp", "segformer"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0"}
GT_COLOR = (60, 220, 60)      # green: reference
PRED_COLOR = (255, 80, 80)    # red: prediction


@torch.no_grad()
def predict(model, img_path, gt, device, size):
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = gt.shape
    x = eval_transform(size)(image=image, mask=gt.astype(np.float32))["image"]
    prob = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].float().cpu().numpy()
    return image, cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def overlay(image, gt, pred=None):
    out = image.copy()
    if pred is not None:
        mask = (pred > 0.5).astype(np.uint8)
        tint = out.copy()
        tint[mask == 1] = PRED_COLOR
        out = cv2.addWeighted(out, 0.62, tint, 0.38, 0)
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cv2.drawContours(out, [c], -1, PRED_COLOR, 2)
    for c in cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        cv2.drawContours(out, [c], -1, GT_COLOR, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "polypseg" / "runs_cv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--data-root", type=Path, default=ROOT / "Kvasir-SEG")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "polypseg" / "outputs" / "figures" / "fig7_qualitative")
    args = ap.parse_args()

    device = pick_device()
    splits = ROOT / "polypseg" / "outputs" / "folds" / f"fold{args.fold}.csv"
    records = kvasir_records(args.data_root, splits, "test")
    meta = pd.read_csv(ROOT / "polypseg" / "outputs" / "cv_assignment.csv")
    meta["sessile"] = meta["sessile"].astype(bool)
    meta = meta.set_index("name")

    models = {}
    for m in MODELS:
        ckpt = args.runs_dir / f"{m}_seed42_fold{args.fold}" / "best.pth"
        net = build_model(m).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        net.eval()
        models[m] = net
    print(f"loaded {len(models)} models on {device}; {len(records)} test images")

    # Score every test image with every model, then pick illustrative cases.
    rows = []
    cache = {}
    for name, img_path, mask_path in records:
        gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        preds = {}
        for m, net in models.items():
            image, prob = predict(net, img_path, gt, device, args.size)
            preds[m] = prob
        cache[name] = (image, gt, preds)
        rows.append(dict(name=name,
                         sessile=bool(meta.loc[name, "sessile"]),
                         area=float(meta.loc[name, "area_frac"]),
                         **{m: region_metrics(preds[m], gt)["dice"] for m in MODELS}))
    df = pd.DataFrame(rows)
    df["mean_dice"] = df[MODELS].mean(axis=1)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    # Two flat-polyp failures, one small-polyp failure, one success for contrast.
    ses_fail = df[df.sessile].nsmallest(2, "mean_dice").name.tolist()
    small_fail = df[(~df.name.isin(ses_fail))].nsmallest(4, "mean_dice")
    small_fail = small_fail[small_fail.area < df.area.median()].name.tolist()[:1]
    success = df[df.sessile].nlargest(1, "mean_dice").name.tolist()
    chosen = ses_fail + small_fail + success
    labels = ["flat, failure", "flat, failure", "small, failure", "flat, success"]
    print("selected:", chosen)

    ncol = 2 + len(MODELS)
    fig, axes = plt.subplots(len(chosen), ncol,
                             figsize=(2.05 * ncol, 1.85 * len(chosen)))
    col_titles = ["Input", "Reference", *[PRETTY[m] for m in MODELS]]
    for r, (name, lab) in enumerate(zip(chosen, labels)):
        image, gt, preds = cache[name]
        panels = [image, overlay(image, gt)] + [overlay(image, gt, preds[m]) for m in MODELS]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
            if c >= 2:
                d = region_metrics(preds[MODELS[c - 2]], gt)["dice"]
                ax.text(0.5, -0.06, f"Dice {d:.3f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8,
                        color="#B00000" if d < 0.5 else "black")
    fig.tight_layout(h_pad=1.3, w_pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
