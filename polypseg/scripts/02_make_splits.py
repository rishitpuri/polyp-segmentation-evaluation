"""Create reproducible train/val/test splits (80/10/10) for Kvasir-SEG,
stratified jointly on polyp size tertile and sessile flag so that every
evaluation stratum is represented proportionally in each split.

Outputs: polypseg/outputs/splits.csv (name, split columns merged into metadata).
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"

df = pd.read_csv(OUT / "metadata.csv")
df["stratum"] = df.size_tertile.astype(str) + "_" + df.sessile.astype(str)

train, tmp = train_test_split(
    df, test_size=0.2, stratify=df.stratum, random_state=SEED
)
val, test = train_test_split(
    tmp, test_size=0.5, stratify=tmp.stratum, random_state=SEED
)

df["split"] = "train"
df.loc[df.name.isin(val.name), "split"] = "val"
df.loc[df.name.isin(test.name), "split"] = "test"
df.drop(columns=["stratum"]).to_csv(OUT / "splits.csv", index=False)

print(df.split.value_counts())
print("\nStrata per split:")
print(df.groupby(["split", "size_tertile"], observed=True).size().unstack())
print("\nSessile per split:")
print(df.groupby("split").sessile.sum())
print(f"\nSaved {OUT / 'splits.csv'}")
