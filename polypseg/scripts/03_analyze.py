"""Aggregate per-image metrics into the paper's main results.

Produces:
  - Table 1: in-domain performance per architecture (mean +- std over seeds)
  - Table 2: stratified performance (size tertile, sessile) with bootstrap CIs
  - Table 3: cross-dataset generalization (Kvasir-SEG -> CVC-ClinicDB)
  - Table 4: effect of test-time augmentation
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
RESULTS = OUT / "analysis"
RESULTS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1337, 2024]
MODELS = ["unet", "unetpp", "segformer"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0"}
METRICS = ["dice", "iou", "boundary_f1", "hd95"]


def load_all():
    frames = []
    for model in MODELS:
        for seed in SEEDS:
            for tta in [False, True]:
                suffix = "_tta" if tta else ""
                path = OUT / f"perimage_{model}_seed{seed}{suffix}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["seed"] = seed
                df["tta"] = tta
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["model"] = pd.Categorical(df.model, categories=MODELS, ordered=True)
    return df


def bootstrap_ci(values, n_boot=10000, seed=0):
    """Percentile bootstrap CI for the mean — appropriate here because the
    per-stratum sample sizes are small (~33 images)."""
    values = np.asarray([v for v in values if not np.isnan(v)])
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_agg(df, group_cols):
    """Mean per seed, then mean +- std across seeds (the correct way to
    report variability: seeds are the unit of replication, not images)."""
    per_seed = df.groupby(group_cols + ["seed"], observed=True)[METRICS].mean().reset_index()
    agg = per_seed.groupby(group_cols, observed=True)[METRICS].agg(["mean", "std"])
    return agg, per_seed


def main():
    df = load_all()
    meta = pd.read_csv(OUT / "splits.csv")[["name", "size_tertile", "area_frac", "sessile", "n_polyps_mask"]]
    df = df.merge(meta, on="name", how="left")

    kv = df[(df.dataset == "kvasir-seg") & (~df.tta)]
    print("=" * 78)
    print("TABLE 1  In-domain (Kvasir-SEG test, n=100), mean +- std over 3 seeds")
    print("=" * 78)
    t1, per_seed = seed_agg(kv, ["model"])
    for model in MODELS:
        r = t1.loc[model]
        print(f"  {PRETTY[model]:14s} Dice {r[('dice','mean')]:.4f}+-{r[('dice','std')]:.4f}  "
              f"IoU {r[('iou','mean')]:.4f}+-{r[('iou','std')]:.4f}  "
              f"BF1 {r[('boundary_f1','mean')]:.4f}  HD95 {r[('hd95','mean')]:.2f}")
    t1.to_csv(RESULTS / "table1_indomain.csv")

    print()
    print("=" * 78)
    print("TABLE 2  Stratified by polyp size tertile (bootstrap 95% CI on pooled seeds)")
    print("=" * 78)
    rows = []
    for model in MODELS:
        for tert in ["small", "medium", "large"]:
            sub = kv[(kv.model == model) & (kv.size_tertile == tert)]
            lo, hi = bootstrap_ci(sub.dice.values)
            rows.append(dict(model=PRETTY[model], stratum=tert, n=len(sub) // len(SEEDS),
                             dice=sub.dice.mean(), ci_lo=lo, ci_hi=hi,
                             boundary_f1=sub.boundary_f1.mean()))
            print(f"  {PRETTY[model]:14s} {tert:7s} n={len(sub)//len(SEEDS):3d}  "
                  f"Dice {sub.dice.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
                  f"BF1 {sub.boundary_f1.mean():.4f}")
    pd.DataFrame(rows).to_csv(RESULTS / "table2_size_strata.csv", index=False)

    print()
    print("  -- Sessile (flat) vs pedunculated --")
    rows = []
    for model in MODELS:
        for flag, label in [(True, "sessile"), (False, "non-sessile")]:
            sub = kv[(kv.model == model) & (kv.sessile == flag)]
            lo, hi = bootstrap_ci(sub.dice.values)
            rows.append(dict(model=PRETTY[model], stratum=label, n=len(sub) // len(SEEDS),
                             dice=sub.dice.mean(), ci_lo=lo, ci_hi=hi))
            print(f"  {PRETTY[model]:14s} {label:12s} n={len(sub)//len(SEEDS):3d}  "
                  f"Dice {sub.dice.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv(RESULTS / "table2b_sessile.csv", index=False)

    print()
    print("=" * 78)
    print("TABLE 3  Cross-dataset generalization (train Kvasir-SEG -> test CVC-ClinicDB)")
    print("=" * 78)
    no_tta = df[~df.tta]
    t3, _ = seed_agg(no_tta, ["model", "dataset"])
    rows = []
    for model in MODELS:
        ind = t3.loc[(model, "kvasir-seg")]
        try:
            ood = t3.loc[(model, "cvc-clinicdb")]
        except KeyError:
            continue
        gap = ind[("dice", "mean")] - ood[("dice", "mean")]
        rel = 100 * gap / ind[("dice", "mean")]
        rows.append(dict(model=PRETTY[model], in_domain=ind[("dice", "mean")],
                         out_domain=ood[("dice", "mean")], gap=gap, relative_drop_pct=rel))
        print(f"  {PRETTY[model]:14s} in-domain {ind[('dice','mean')]:.4f}  "
              f"CVC-ClinicDB {ood[('dice','mean')]:.4f}+-{ood[('dice','std')]:.4f}  "
              f"gap {gap:+.4f} ({rel:.1f}% relative drop)")
    pd.DataFrame(rows).to_csv(RESULTS / "table3_cross_dataset.csv", index=False)

    print()
    print("=" * 78)
    print("TABLE 4  Effect of test-time augmentation (Dice)")
    print("=" * 78)
    t4, _ = seed_agg(df, ["model", "dataset", "tta"])
    for model in MODELS:
        for ds in sorted(df.dataset.unique()):
            try:
                base = t4.loc[(model, ds, False)][("dice", "mean")]
                with_tta = t4.loc[(model, ds, True)][("dice", "mean")]
            except KeyError:
                continue
            print(f"  {PRETTY[model]:14s} {ds:14s} {base:.4f} -> {with_tta:.4f} "
                  f"({with_tta - base:+.4f})")
    t4.to_csv(RESULTS / "table4_tta.csv")

    df.to_csv(RESULTS / "all_perimage_merged.csv", index=False)
    print(f"\nWrote tables to {RESULTS}")


if __name__ == "__main__":
    main()
