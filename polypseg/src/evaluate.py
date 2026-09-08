"""Evaluate a trained checkpoint and write per-image metrics.

Predictions are resized back to each image's native resolution and scored
against the original mask, so scores are not inflated by evaluating on a
downsampled grid. Per-image rows let us aggregate by stratum (size tertile,
sessile) afterwards without re-running inference.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD, eval_transform, folder_records, kvasir_records
from .metrics import all_metrics
from .models import MODELS, build_model
from .train import pick_device

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def predict_records(model, records, device, size, tta=False):
    """Yield (name, prob_map_at_native_resolution, gt_at_native_resolution)."""
    tf = eval_transform(size)
    for name, img_path, mask_path in records:
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        h, w = gt.shape
        x = tf(image=image, mask=gt.astype(np.float32))["image"].unsqueeze(0).to(device)

        logits = model(x)
        prob = torch.sigmoid(logits)
        if tta:
            for dims in ([3], [2], [2, 3]):
                flipped = torch.flip(x, dims=dims)
                prob = prob + torch.flip(torch.sigmoid(model(flipped)), dims=dims)
            prob = prob / 4.0

        prob = prob[0, 0].float().cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        yield name, prob, gt


def evaluate(model, records, device, size, dataset_name, tta=False):
    rows = []
    for name, prob, gt in predict_records(model, records, device, size, tta):
        m = all_metrics(prob, gt)
        m.update(name=name, dataset=dataset_name)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true", help="4-way flip test-time augmentation")
    ap.add_argument("--data-root", type=Path, default=ROOT / "Kvasir-SEG")
    ap.add_argument("--splits", type=Path, default=ROOT / "polypseg" / "outputs" / "splits.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--external", nargs="*", default=[],
                    help="Cross-dataset test sets as name=/path/to/root (expects images/ and masks/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = pick_device(args.device)
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frames = [evaluate(model, kvasir_records(args.data_root, args.splits, args.split),
                       device, args.size, "kvasir-seg", args.tta)]

    for spec in args.external:
        name, _, path = spec.partition("=")
        root = Path(path)
        records = folder_records(root / "images", root / "masks")
        print(f"[eval] {name}: {len(records)} images", flush=True)
        frames.append(evaluate(model, records, device, args.size, name, args.tta))

    df = pd.concat(frames, ignore_index=True)
    df["model"] = args.model
    df["checkpoint"] = str(args.checkpoint)
    df["tta"] = args.tta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.groupby("dataset")[["dice", "iou", "boundary_f1"]].mean().round(4))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
