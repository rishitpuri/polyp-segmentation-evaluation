"""Compute per-image ground-truth statistics for the external test sets.

Kvasir-SEG metadata comes from 01_data_audit.py. The external datasets need
the same treatment so that size-stratified analysis can be run out of domain
-- which matters because CVC-ColonDB and ETIS-LaribPolypDB contain lesions an
order of magnitude smaller than anything in Kvasir-SEG.

Output: polypseg/outputs/external_metadata.csv
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = ROOT / "polypseg" / "outputs"

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    rows = []
    for d in sorted(EXT.iterdir()):
        if not (d / "masks").is_dir():
            continue
        for mp in sorted((d / "masks").iterdir()):
            if mp.suffix.lower() not in IMG_EXT:
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            binary = (mask > 127).astype(np.uint8)
            h, w = binary.shape
            n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = areas[areas > 2e-4 * h * w]
            rows.append(dict(
                dataset=d.name, name=mp.stem, height=h, width=w,
                area_frac=float(binary.sum()) / (h * w),
                n_lesions=int(len(keep)),
                largest_frac=float(keep.max()) / (h * w) if len(keep) else 0.0,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "external_metadata.csv", index=False)

    print("Per-dataset ground-truth statistics")
    print(f"{'dataset':<16}{'n':>5}{'median area%':>14}{'p10 area%':>11}"
          f"{'p90 area%':>11}{'multi-lesion':>14}")
    for ds, g in df.groupby("dataset"):
        print(f"{ds:<16}{len(g):>5}{100 * g.area_frac.median():>14.2f}"
              f"{100 * g.area_frac.quantile(0.1):>11.2f}"
              f"{100 * g.area_frac.quantile(0.9):>11.2f}"
              f"{100 * (g.n_lesions > 1).mean():>13.1f}%")

    # Kvasir for comparison
    kv = pd.read_csv(OUT / "metadata.csv")
    print(f"{'kvasir-seg':<16}{len(kv):>5}{100 * kv.area_frac.median():>14.2f}"
          f"{100 * kv.area_frac.quantile(0.1):>11.2f}"
          f"{100 * kv.area_frac.quantile(0.9):>11.2f}"
          f"{100 * (kv.n_polyps_mask > 1).mean():>13.1f}%")

    print(f"\nFraction of images with polyp area < 1% (clinically small):")
    for ds, g in df.groupby("dataset"):
        print(f"  {ds:<16}{100 * (g.area_frac < 0.01).mean():>6.1f}%")
    print(f"  {'kvasir-seg':<16}{100 * (kv.area_frac < 0.01).mean():>6.1f}%")
    print(f"\nWrote {OUT / 'external_metadata.csv'}")


if __name__ == "__main__":
    main()
