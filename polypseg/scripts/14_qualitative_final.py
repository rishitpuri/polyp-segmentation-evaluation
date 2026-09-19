"""Qualitative panel for the manuscript's central claim.

Shows, for all five architectures, cases drawn from the *external* datasets
where lesions occupy less than 1% of the frame -- the size range in which
per-lesion detection collapses to roughly one in two. External images are
never seen during training, so every prediction shown is out-of-sample.

Layout: input | reference | one column per architecture, annotated with Dice
and whether the lesion was detected at IoU >= 0.5.

Run on a GPU host after training fold-0 models for all five architectures.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from polypseg.src.data import eval_transform, folder_records
from polypseg.src.metrics import lesion_metrics, region_metrics
from polypseg.src.models import MODELS, build_model
from polypseg.src.train import pick_device

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["unet", "unetpp", "segformer", "segformer_b2", "unet_pvt"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0",
          "segformer_b2": "SegFormer-B2", "unet_pvt": "U-Net+PVTv2"}
GT_COLOR = (60, 220, 60)
PRED_COLOR = (255, 80, 80)


@torch.no_grad()
def predict(model, img_path, gt, device, size):
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = gt.shape
    x = eval_transform(size)(image=image, mask=gt.astype(np.float32))["image"]
    prob = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].float().cpu().numpy()
    return image, cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_around_lesion(gt, pad_frac=0.35):
    """Small lesions are invisible in a full frame; crop generously around the
    reference so the reader can actually see what is being missed."""
    ys, xs = np.where(gt > 0)
    if len(ys) == 0:
        return 0, gt.shape[0], 0, gt.shape[1]
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = gt.shape
    side = max(y1 - y0, x1 - x0)
    pad = int(max(side * pad_frac, 0.08 * min(h, w)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2 + pad
    return (max(0, cy - half), min(h, cy + half),
            max(0, cx - half), min(w, cx + half))


def overlay(image, gt, pred=None):
    out = image.copy()
    if pred is not None:
        mask = (pred > 0.5).astype(np.uint8)
        tint = out.copy()
        tint[mask == 1] = PRED_COLOR
        out = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cv2.drawContours(out, [c], -1, PRED_COLOR, 2)
    for c in cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        cv2.drawContours(out, [c], -1, GT_COLOR, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "polypseg" / "runs_qual")
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--max-area", type=float, default=0.01)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "polypseg" / "outputs" / "figures_final" / "f8_qualitative")
    args = ap.parse_args()

    device = pick_device()
    meta = pd.read_csv(ROOT / "polypseg" / "outputs" / "external_metadata.csv")
    meta["name"] = meta["name"].astype(str)
    small = meta[meta.area_frac < args.max_area]
    print(f"{len(small)} external images with lesion area < {args.max_area:.0%}")

    models = {}
    for m in ORDER:
        ckpt = args.runs_dir / f"{m}_seed42_qual" / "best.pth"
        net = build_model(m).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        net.eval()
        models[m] = net
    print(f"loaded {len(models)} models on {device}")

    # Score every small-lesion external image with every architecture.
    rows, cache = [], {}
    for _, r in small.iterrows():
        d = ROOT / "data" / "external" / r.dataset
        recs = {n: (ip, mp) for n, ip, mp in folder_records(d / "images", d / "masks")}
        if r["name"] not in recs:
            continue
        ip, mp = recs[r["name"]]
        gt = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        preds, dices, dets = {}, {}, {}
        for m, net in models.items():
            image, prob = predict(net, ip, gt, device, args.size)
            preds[m] = prob
            dices[m] = region_metrics(prob, gt)["dice"]
            dets[m] = lesion_metrics(prob, gt)["det_rate_iou05"]
        key = (r.dataset, r["name"])
        cache[key] = (image, gt, preds, dices, dets)
        rows.append(dict(dataset=r.dataset, name=r["name"], area=r.area_frac,
                         mean_dice=float(np.mean(list(dices.values()))),
                         mean_det=float(np.nanmean(list(dets.values())))))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"scored {len(df)} images")

    # Three total failures plus one success, for contrast.
    fails = df.nsmallest(3, "mean_dice")
    success = df.nlargest(1, "mean_dice")
    chosen = list(fails.itertuples()) + list(success.itertuples())

    ncol = 2 + len(ORDER)
    fig, axes = plt.subplots(len(chosen), ncol,
                             figsize=(1.55 * ncol, 1.72 * len(chosen)))
    titles = ["Input", "Reference"] + [PRETTY[m] for m in ORDER]
    for ri, rec in enumerate(chosen):
        image, gt, preds, dices, dets = cache[(rec.dataset, rec.name)]
        y0, y1, x0, x1 = crop_around_lesion(gt)
        panels = [image, overlay(image, gt)] + [overlay(image, gt, preds[m]) for m in ORDER]
        for ci, panel in enumerate(panels):
            ax = axes[ri, ci]
            ax.imshow(panel[y0:y1, x0:x1])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(titles[ci], fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{rec.dataset}\n{100 * rec.area:.2f}% area", fontsize=6.5)
            if ci >= 2:
                m = ORDER[ci - 2]
                d = dices[m]
                ax.text(0.5, -0.04, f"{d:.2f}", transform=ax.transAxes, ha="center",
                        va="top", fontsize=7.5,
                        color="#B00000" if d < 0.5 else "black")
    fig.tight_layout(h_pad=1.1, w_pad=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}.pdf/.png")
    print("\nselected cases:")
    for rec in chosen:
        print(f"  {rec.dataset:14s} {rec.name:>6s}  area={100 * rec.area:5.2f}%  "
              f"mean Dice={rec.mean_dice:.3f}  mean det={rec.mean_det:.2f}")


if __name__ == "__main__":
    main()
