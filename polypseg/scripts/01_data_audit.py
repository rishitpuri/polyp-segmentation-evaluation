"""Data audit for Kvasir-SEG: verify image/mask pairs, compute per-image
polyp statistics, and flag sessile (flat) polyps via the Kvasir-Sessile subset.

Outputs: polypseg/outputs/metadata.csv and a printed summary.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KVASIR = ROOT / "Kvasir-SEG"
SESSILE = ROOT / "sessile-main-Kvasir-SEG"
OUT = ROOT / "polypseg" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def audit():
    images = sorted((KVASIR / "images").glob("*.jpg"))
    masks = sorted((KVASIR / "masks").glob("*.jpg"))
    img_names = {p.stem for p in images}
    mask_names = {p.stem for p in masks}
    assert img_names == mask_names, (
        f"Mismatch: {img_names ^ mask_names}"
    )
    sessile_names = {p.stem for p in (SESSILE / "images").glob("*.jpg")}
    print(f"Kvasir-SEG pairs: {len(images)}")
    print(f"Sessile subset:   {len(sessile_names)}")
    print(f"Sessile names found in main set: {len(sessile_names & img_names)}")

    bboxes = json.loads((KVASIR / "kavsir_bboxes.json").read_text())

    rows = []
    for img_path in images:
        name = img_path.stem
        mask = cv2.imread(str(KVASIR / "masks" / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_path))
        h, w = mask.shape
        binary = (mask > 127).astype(np.uint8)
        area_frac = float(binary.sum()) / (h * w)
        n_comp, _, comp_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        comp_areas = comp_stats[1:, cv2.CC_STAT_AREA]
        n_polyps_mask = int((comp_areas > 0.0005 * h * w).sum())
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        area_px = binary.sum()
        compactness = (4 * np.pi * area_px / perimeter**2) if perimeter > 0 else 0.0
        n_bbox = len(bboxes.get(name, {}).get("bbox", []))
        rows.append(dict(
            name=name, height=h, width=w,
            img_h=img.shape[0], img_w=img.shape[1],
            area_frac=area_frac,
            n_polyps_mask=n_polyps_mask,
            n_bbox=n_bbox,
            compactness=float(compactness),
            sessile=name in sessile_names,
        ))

    df = pd.DataFrame(rows)

    # Size strata based on relative polyp area (common convention:
    # small < 1%? too strict for Kvasir; use tertile-informed clinical bins)
    df["size_bin"] = pd.cut(
        df.area_frac, bins=[0, 0.02, 0.10, 1.0],
        labels=["small", "medium", "large"],
    )
    # Quantile tertiles: balanced strata for splitting / stratified evaluation
    df["size_tertile"] = pd.qcut(
        df.area_frac, q=3, labels=["small", "medium", "large"]
    )
    df.to_csv(OUT / "metadata.csv", index=False)

    print("\n--- Summary ---")
    print(df.area_frac.describe())
    print("\nSize bins (area fraction: <2% small, 2-10% medium, >10% large):")
    print(df.size_bin.value_counts())
    print("\nSize tertiles (quantile cut points):")
    print(df.groupby("size_tertile", observed=True).area_frac.agg(["min", "max", "count"]))
    print("\nSessile overlap by size bin:")
    print(df.groupby("size_bin", observed=True).sessile.agg(["sum", "count"]))
    print("\nMulti-polyp images (mask components):", (df.n_polyps_mask > 1).sum())
    print("bbox vs mask count disagreement:", (df.n_polyps_mask != df.n_bbox).sum())
    print("\nImage size mismatches (img vs mask):",
          ((df.height != df.img_h) | (df.width != df.img_w)).sum())
    print(f"\nSaved {OUT / 'metadata.csv'}")


if __name__ == "__main__":
    audit()
