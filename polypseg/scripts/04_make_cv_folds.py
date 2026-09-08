"""Create 5-fold cross-validation splits over all 1000 Kvasir-SEG images.

Motivation: the single 80/10/10 split leaves only ~33 images per size
tertile and 19 sessile images in the test set, so per-stratum confidence
intervals are too wide to support claims. Under 5-fold CV every image is
tested exactly once, giving n=1000 overall, ~333 per tertile and n=196 for
the sessile stratum.

For each fold k: test = fold k (200 images); a stratified 100-image
validation set is carved from the remaining 800 for early stopping; the
other 700 are used for training.

Each fold is written as a standalone splits-style CSV so train.py and
evaluate.py work unchanged (they just take a different --splits file).
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
N_FOLDS = 5
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
FOLD_DIR = OUT / "folds"
FOLD_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "metadata.csv")
df["stratum"] = df.size_tertile.astype(str) + "_" + df.sessile.astype(str)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df["fold"] = -1
for k, (_, test_idx) in enumerate(skf.split(df, df.stratum)):
    df.loc[df.index[test_idx], "fold"] = k
assert (df.fold >= 0).all()

df.to_csv(OUT / "cv_assignment.csv", index=False)

for k in range(N_FOLDS):
    fold = df.copy()
    test_mask = fold.fold == k
    remaining = fold[~test_mask]
    train_part, val_part = train_test_split(
        remaining, test_size=100, stratify=remaining.stratum, random_state=SEED
    )
    fold["split"] = "train"
    fold.loc[test_mask, "split"] = "test"
    fold.loc[fold.name.isin(val_part.name), "split"] = "val"
    fold.drop(columns=["stratum"]).to_csv(FOLD_DIR / f"fold{k}.csv", index=False)
    counts = fold.split.value_counts()
    print(f"fold{k}: train={counts['train']} val={counts['val']} test={counts['test']}")

print("\nTest-set stratum coverage across folds (each image tested exactly once):")
print(df.groupby(["fold", "size_tertile"], observed=True).size().unstack())
print("\nSessile per fold test set:")
print(df.groupby("fold").sessile.sum())
print(f"\nTotals across all folds -> tertiles: "
      f"{df.size_tertile.value_counts().to_dict()}, sessile: {int(df.sessile.sum())}")
print(f"\nWrote {N_FOLDS} fold files to {FOLD_DIR}")
