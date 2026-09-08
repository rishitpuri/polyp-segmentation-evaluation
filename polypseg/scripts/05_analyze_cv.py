"""Analysis of the 5-fold cross-validation results.

Because every image is tested exactly once across folds, per-stratum
estimates use n~333 (size tertiles) and n=196 (sessile) rather than the
33/19 available from a single split.

Statistics:
  - Architecture comparisons use a paired Wilcoxon signed-rank test over the
    1000 per-image Dice scores (paired because all models see the same
    images), with Holm correction over the 3 pairwise comparisons.
  - Stratum contrasts (small vs large, sessile vs not) use Mann-Whitney U
    (independent groups) plus a bootstrap CI on the difference of means.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
CV = OUT / "cv"
RESULTS = OUT / "analysis_cv"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "unetpp", "segformer"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0"}
FOLDS = range(5)


def load(tta=False):
    frames = []
    for model in MODELS:
        for k in FOLDS:
            suffix = "_tta" if tta else ""
            path = CV / f"perimage_{model}_seed42_fold{k}{suffix}.csv"
            df = pd.read_csv(path)
            df["fold"] = k
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def boot_diff(a, b, n_boot=10000, seed=0):
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    da = rng.choice(a, (n_boot, len(a)), replace=True).mean(1)
    db = rng.choice(b, (n_boot, len(b)), replace=True).mean(1)
    d = da - db
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (len(pvals) - rank))
    return np.maximum.accumulate(adj[order])[np.argsort(order)]


def main():
    df = load(tta=False)
    tta = load(tta=True)
    meta = pd.read_csv(OUT / "cv_assignment.csv")[
        ["name", "size_tertile", "area_frac", "sessile", "n_polyps_mask"]]
    df = df.merge(meta, on="name", how="left")
    tta = tta.merge(meta, on="name", how="left")

    kv = df[df.dataset == "kvasir-seg"].copy()
    ood = df[df.dataset == "cvc-clinicdb"].copy()
    assert len(kv) == 3000, f"expected 3 models x 1000 images, got {len(kv)}"

    # The merge introduces NaN for external-dataset rows, which makes the
    # sessile column object-dtype. `~series` on object dtype yields integers
    # (-1/-2), both of which are truthy, so a naive `~kv.sessile` mask silently
    # selects every row. Cast explicitly and assert the partition is exact.
    kv["sessile"] = kv["sessile"].astype(bool)
    n_ses = int(kv[kv.model == MODELS[0]].sessile.sum())
    n_non = int((~kv[kv.model == MODELS[0]].sessile).sum())
    assert n_ses == 196 and n_non == 804, f"bad sessile partition: {n_ses}/{n_non}"

    print("=" * 82)
    print("TABLE 1  In-domain, 5-fold CV (every image tested once, n=1000 per model)")
    print("=" * 82)
    rows = []
    for m in MODELS:
        s = kv[kv.model == m]
        per_fold = s.groupby("fold").dice.mean()
        rows.append(dict(model=PRETTY[m], dice=s.dice.mean(), fold_std=per_fold.std(),
                         iou=s.iou.mean(), boundary_f1=s.boundary_f1.mean(),
                         hd95_pct=s.hd95_pct_diag.mean()))
        print(f"  {PRETTY[m]:14s} Dice {s.dice.mean():.4f} (fold sd {per_fold.std():.4f})  "
              f"IoU {s.iou.mean():.4f}  BF1 {s.boundary_f1.mean():.4f}  "
              f"HD95 {s.hd95_pct_diag.mean():.2f}% diag")
    pd.DataFrame(rows).to_csv(RESULTS / "cv_table1_indomain.csv", index=False)

    print("\n  Paired Wilcoxon on 1000 per-image Dice scores (Holm-corrected):")
    pairs, praw = [], []
    for a, b in combinations(MODELS, 2):
        xa = kv[kv.model == a].set_index("name").dice
        xb = kv[kv.model == b].set_index("name").dice
        common = xa.index.intersection(xb.index)
        stat, p = wilcoxon(xa.loc[common], xb.loc[common])
        pairs.append((a, b, xa.loc[common].mean() - xb.loc[common].mean()))
        praw.append(p)
    padj = holm(praw)
    for (a, b, d), p, pa in zip(pairs, praw, padj):
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "ns"
        print(f"    {PRETTY[a]:14s} vs {PRETTY[b]:14s} dDice {d:+.4f}  "
              f"p={p:.2e}  p_holm={pa:.2e}  {sig}")

    print()
    print("=" * 82)
    print("TABLE 2  Size strata (n~333 each)")
    print("=" * 82)
    rows = []
    for m in MODELS:
        line = []
        for t in ["small", "medium", "large"]:
            s = kv[(kv.model == m) & (kv.size_tertile == t)]
            line.append(f"{t} {s.dice.mean():.4f} (n={len(s)})")
            rows.append(dict(model=PRETTY[m], stratum=t, n=len(s), dice=s.dice.mean(),
                             boundary_f1=s.boundary_f1.mean()))
        print(f"  {PRETTY[m]:14s} " + "  ".join(line))
    print("\n  small vs large contrast:")
    for m in MODELS:
        sm = kv[(kv.model == m) & (kv.size_tertile == "small")].dice.values
        lg = kv[(kv.model == m) & (kv.size_tertile == "large")].dice.values
        d, lo, hi = boot_diff(sm, lg)
        _, p = mannwhitneyu(sm, lg)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    {PRETTY[m]:14s} dDice {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"p={p:.2e}  {sig}")
    pd.DataFrame(rows).to_csv(RESULTS / "cv_table2_size.csv", index=False)

    print()
    print("=" * 82)
    print("TABLE 3  Sessile (flat, n=196) vs non-sessile (n=804)  <-- key claim")
    print("=" * 82)
    rows = []
    for m in MODELS:
        ses = kv[(kv.model == m) & (kv.sessile)].dice.values
        non = kv[(kv.model == m) & (~kv.sessile)].dice.values
        d, lo, hi = boot_diff(ses, non)
        _, p = mannwhitneyu(ses, non)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rows.append(dict(model=PRETTY[m], sessile=ses.mean(), non_sessile=non.mean(),
                         diff=d, ci_lo=lo, ci_hi=hi, p=p))
        print(f"  {PRETTY[m]:14s} sessile {ses.mean():.4f}  non-sessile {non.mean():.4f}  "
              f"d {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.2e}  {sig}")
    pd.DataFrame(rows).to_csv(RESULTS / "cv_table3_sessile.csv", index=False)

    print("\n  Between-model comparison ON SESSILE ONLY (paired Wilcoxon, Holm):")
    pairs, praw = [], []
    ses_only = kv[kv.sessile]
    for a, b in combinations(MODELS, 2):
        xa = ses_only[ses_only.model == a].set_index("name").dice
        xb = ses_only[ses_only.model == b].set_index("name").dice
        common = xa.index.intersection(xb.index)
        _, p = wilcoxon(xa.loc[common], xb.loc[common])
        pairs.append((a, b, xa.loc[common].mean() - xb.loc[common].mean(), len(common)))
        praw.append(p)
    padj = holm(praw)
    for (a, b, d, n), p, pa in zip(pairs, praw, padj):
        sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "ns"
        print(f"    {PRETTY[a]:14s} vs {PRETTY[b]:14s} dDice {d:+.4f} (n={n})  "
              f"p_holm={pa:.2e}  {sig}")

    print()
    print("=" * 82)
    print("TABLE 4  Cross-dataset: Kvasir-SEG -> CVC-ClinicDB (5 independent fold models)")
    print("=" * 82)
    rows = []
    for m in MODELS:
        i = kv[kv.model == m].dice.mean()
        o = ood[ood.model == m]
        per_fold = o.groupby("fold").dice.mean()
        rows.append(dict(model=PRETTY[m], in_domain=i, out_domain=per_fold.mean(),
                         out_fold_std=per_fold.std(),
                         gap=i - per_fold.mean(), rel_drop=100 * (i - per_fold.mean()) / i))
        print(f"  {PRETTY[m]:14s} in {i:.4f} -> CVC {per_fold.mean():.4f} "
              f"(fold sd {per_fold.std():.4f})  gap {i - per_fold.mean():+.4f} "
              f"({100 * (i - per_fold.mean()) / i:.1f}% rel)")
    pd.DataFrame(rows).to_csv(RESULTS / "cv_table4_cross_dataset.csv", index=False)

    print()
    print("=" * 82)
    print("TABLE 5  Test-time augmentation")
    print("=" * 82)
    for m in MODELS:
        for ds in ["kvasir-seg", "cvc-clinicdb"]:
            b = df[(df.model == m) & (df.dataset == ds)].dice.mean()
            t = tta[(tta.model == m) & (tta.dataset == ds)].dice.mean()
            print(f"  {PRETTY[m]:14s} {ds:14s} {b:.4f} -> {t:.4f} ({t - b:+.4f})")

    df.to_csv(RESULTS / "cv_perimage_merged.csv", index=False)
    tta.to_csv(RESULTS / "cv_perimage_merged_tta.csv", index=False)
    print(f"\nWrote tables to {RESULTS}")


if __name__ == "__main__":
    main()
