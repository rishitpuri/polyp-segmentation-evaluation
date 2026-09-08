"""Final analysis for the journal manuscript.

Consolidates every experiment:
  - 5-fold cross-validation, 5 architectures, in-domain (Kvasir-SEG)
  - zero-shot transfer to 4 external datasets
  - stratified performance in and out of domain
  - per-lesion detection
  - the measured run-to-run noise floor, and which architecture gaps exceed it
  - the single-split vs cross-validation replication failure

Writes CSVs to polypseg/outputs/analysis_final/ and prints a readable report.
"""
import itertools
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
RES = OUT / "analysis_final"
RES.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "unetpp", "segformer", "segformer_b2", "unet_pvt"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0",
          "segformer_b2": "SegFormer-B2", "unet_pvt": "U-Net+PVTv2"}
PARAMS = {"unet": 24.44, "unetpp": 26.08, "segformer": 3.71,
          "segformer_b2": 24.72, "unet_pvt": 28.13}
DATASETS = ["kvasir-seg", "cvc-clinicdb", "cvc-300", "cvc-colondb", "etis-larib"]
DSNAME = {"kvasir-seg": "Kvasir-SEG", "cvc-clinicdb": "CVC-ClinicDB",
          "cvc-300": "CVC-300", "cvc-colondb": "CVC-ColonDB",
          "etis-larib": "ETIS-Larib"}


def boot_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray([x for x in a if not np.isnan(x)])
    b = np.asarray([x for x in b if not np.isnan(x)])
    d = (rng.choice(a, (n_boot, len(a)), replace=True).mean(1)
         - rng.choice(b, (n_boot, len(b)), replace=True).mean(1))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def perm_test(a, b, n_perm=20000, seed=0):
    """Two-sided permutation test on the difference of means.

    Used in preference to Mann-Whitney U so that the p-value and the bootstrap
    confidence interval refer to the same estimand. The two disagree on these
    skewed Dice distributions: a rank-based test can be highly significant
    while the CI for the mean difference spans zero.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray([x for x in a if not np.isnan(x)])
    b = np.asarray([x for x in b if not np.isnan(x)])
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, p[idx] * (len(p) - rank))
    return np.maximum.accumulate(adj[order])[np.argsort(order)]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def load_cv(tta=False):
    frames = []
    for m in MODELS:
        for k in range(5):
            suf = "_tta" if tta else ""
            p = OUT / "cv2" / f"perimage_{m}_seed42_fold{k}{suf}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d["fold"] = k
                frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_repeats():
    """Per-run in-domain test Dice for every repeat condition."""
    recs = []
    for rd, od in [("runs_repeat", "repeat"), ("runs_repeat2", "repeat2")]:
        for f in glob.glob(str(ROOT / "polypseg" / rd / "*" / "summary.json")):
            s = json.load(open(f))
            csv = OUT / od / f"perimage_{s['run']}.csv"
            if not csv.exists():
                continue
            d = pd.read_csv(csv)
            d = d[d.dataset == "kvasir-seg"]
            tag = s["run"]
            cond = ("A" if "fold0rep" in tag else "B" if "fold0det" in tag
                    else "C" if "fixC" in tag else "D")
            recs.append(dict(model=s["model"], cond=cond, run=tag,
                             epoch=s["best_epoch"], val=s["best_val_dice"],
                             test=d.dice.mean()))
    return pd.DataFrame(recs)


def main():
    cv = load_cv()
    tta = load_cv(tta=True)
    kv_meta = pd.read_csv(OUT / "cv_assignment.csv")[
        ["name", "size_tertile", "area_frac", "sessile", "n_polyps_mask"]]
    ext_meta = pd.read_csv(OUT / "external_metadata.csv")[
        ["dataset", "name", "area_frac", "n_lesions"]]

    kv = cv[cv.dataset == "kvasir-seg"].merge(kv_meta, on="name", how="left")
    kv["sessile"] = kv["sessile"].astype(bool)
    assert len(kv) == 5000, len(kv)

    # External dataset filenames are numeric ("1", "10", ...) so pandas infers
    # int64 in one CSV and object in another; force a common type before merging.
    ext_left = cv[cv.dataset != "kvasir-seg"].copy()
    ext_left["name"] = ext_left["name"].astype(str)
    ext_meta["name"] = ext_meta["name"].astype(str)
    ext = ext_left.merge(ext_meta, on=["dataset", "name"], how="left",
                         suffixes=("", "_gt"))
    unmatched = int(ext.area_frac.isna().sum())
    assert unmatched == 0, f"{unmatched} external rows failed to match metadata"

    rep = load_repeats()
    sigma_run = rep[rep.cond.isin(["A", "C", "D"])].groupby(
        ["model", "cond"]).test.std().mean()

    print("=" * 84)
    print("TABLE 1  In-domain, 5-fold CV (n=1000 per model, every image tested once)")
    print("=" * 84)
    rows = []
    for m in MODELS:
        s = kv[kv.model == m]
        rows.append(dict(model=PRETTY[m], params_M=PARAMS[m], dice=s.dice.mean(),
                         iou=s.iou.mean(), bf1=s.boundary_f1.mean(),
                         hd95_pct=s.hd95_pct_diag.mean(),
                         det_iou50=s.det_rate_iou05.mean(),
                         fp_per_img=s.n_false_positives.mean()))
        print(f"  {PRETTY[m]:<14}{PARAMS[m]:>6.1f}M  Dice {s.dice.mean():.4f}  "
              f"IoU {s.iou.mean():.4f}  BF1 {s.boundary_f1.mean():.4f}  "
              f"HD95 {s.hd95_pct_diag.mean():.2f}%  det@.5 {s.det_rate_iou05.mean():.4f}")
    pd.DataFrame(rows).to_csv(RES / "t1_indomain.csv", index=False)

    print()
    print("=" * 84)
    print("TABLE 2  Zero-shot transfer (Dice; mean of 5 fold-models)")
    print("=" * 84)
    hdr = "".join(f"{DSNAME[d][:12]:>13}" for d in DATASETS)
    print(f"  {'model':<14}{hdr}")
    rows = []
    for m in MODELS:
        vals = []
        for ds in DATASETS:
            s = cv[(cv.model == m) & (cv.dataset == ds)]
            vals.append(s.groupby("fold").dice.mean().mean())
        rows.append(dict(model=PRETTY[m], **{DSNAME[d]: v for d, v in zip(DATASETS, vals)}))
        print(f"  {PRETTY[m]:<14}" + "".join(f"{v:>13.4f}" for v in vals))
    pd.DataFrame(rows).to_csv(RES / "t2_cross_dataset.csv", index=False)

    print()
    print("  Per-lesion detection rate @ IoU 0.5")
    rows = []
    for m in MODELS:
        vals = [cv[(cv.model == m) & (cv.dataset == ds)].det_rate_iou05.mean()
                for ds in DATASETS]
        rows.append(dict(model=PRETTY[m], **{DSNAME[d]: v for d, v in zip(DATASETS, vals)}))
        print(f"  {PRETTY[m]:<14}" + "".join(f"{v:>13.4f}" for v in vals))
    pd.DataFrame(rows).to_csv(RES / "t2b_detection.csv", index=False)

    print()
    print("=" * 84)
    print("TABLE 3  Resolving power: architecture spread vs the noise floor")
    print("=" * 84)
    print(f"  measured sigma_run = {sigma_run:.4f} "
          f"(sd across identical repeated runs, {len(rep)} runs)")
    rows = []
    for ds in DATASETS:
        means = {m: cv[(cv.model == m) & (cv.dataset == ds)]
                 .groupby("fold").dice.mean().mean() for m in MODELS}
        best, worst = max(means, key=means.get), min(means, key=means.get)
        spread = means[best] - means[worst]
        rows.append(dict(dataset=DSNAME[ds], spread=spread,
                         in_sigma=spread / sigma_run,
                         best=PRETTY[best], worst=PRETTY[worst]))
        print(f"  {DSNAME[ds]:<14} spread {spread:.4f}  = {spread / sigma_run:5.1f} sigma_run"
              f"   best {PRETTY[best]:<13} worst {PRETTY[worst]}")
    pd.DataFrame(rows).to_csv(RES / "t3_resolving_power.csv", index=False)

    print()
    print("  Pairwise in-domain gaps against the noise floor")
    rows = []
    kvm = {m: kv[kv.model == m].groupby("fold").dice.mean().mean() for m in MODELS}
    for a, b in itertools.combinations(MODELS, 2):
        gap = abs(kvm[a] - kvm[b])
        rows.append(dict(a=PRETTY[a], b=PRETTY[b], gap=gap, ratio=gap / sigma_run,
                         resolvable=gap / sigma_run >= 2))
        flag = "resolvable" if gap / sigma_run >= 2 else "BELOW NOISE"
        print(f"    {PRETTY[a]:<14} vs {PRETTY[b]:<14} {gap:.4f}  "
              f"{gap / sigma_run:4.1f} sigma   {flag}")
    pd.DataFrame(rows).to_csv(RES / "t3b_pairwise.csv", index=False)

    print()
    print("=" * 84)
    print("TABLE 4  Run-to-run variance by condition (identical configs, fold 0)")
    print("=" * 84)
    labels = {"A": "unfixed worker RNG, default kernels",
              "B": "unfixed worker RNG, deterministic",
              "C": "fixed worker RNG, default kernels",
              "D": "fixed worker RNG, deterministic"}
    rows = []
    for c in ["A", "B", "C", "D"]:
        g = rep[rep.cond == c]
        if g.empty:
            continue
        spreads = g.groupby("model").test.agg(lambda s: s.max() - s.min())
        sds = g.groupby("model").test.std()
        rows.append(dict(condition=c, description=labels[c], n_models=len(spreads),
                         mean_sd=sds.mean(), mean_spread=spreads.mean(),
                         max_spread=spreads.max()))
        print(f"  {c}  {labels[c]:<40} mean sd {sds.mean():.4f}  "
              f"mean spread {spreads.mean():.4f}  max {spreads.max():.4f}")
    pd.DataFrame(rows).to_csv(RES / "t4_variance_conditions.csv", index=False)

    print()
    print("  Mechanism: checkpoint selection instability")
    pairs = []
    for (m, c), g in rep.groupby(["model", "cond"]):
        for x, y in itertools.combinations(g.to_dict("records"), 2):
            pairs.append(dict(d_epoch=abs(x["epoch"] - y["epoch"]),
                              d_val=abs(x["val"] - y["val"]),
                              d_test=abs(x["test"] - y["test"])))
    P = pd.DataFrame(pairs)
    r_ep = P.d_epoch.corr(P.d_test)
    r_val = P.d_val.corr(P.d_test)
    close, far = P[P.d_epoch <= 5], P[P.d_epoch >= 15]
    print(f"    {len(P)} pairs of identical runs; mean |delta epoch| = {P.d_epoch.mean():.1f}")
    print(f"    corr(|d epoch|, |d test Dice|) = {r_ep:+.3f}")
    print(f"    corr(|d val Dice|, |d test Dice|) = {r_val:+.3f}   <- val differences "
          f"carry no information about test differences")
    print(f"    |d test| when epochs <=5 apart : {close.d_test.mean():.4f} (n={len(close)})")
    print(f"    |d test| when epochs >=15 apart: {far.d_test.mean():.4f} (n={len(far)})")
    P.to_csv(RES / "t4b_selection_pairs.csv", index=False)
    json.dump(dict(sigma_run=float(sigma_run), r_epoch_test=float(r_ep),
                   r_val_test=float(r_val), mean_d_epoch=float(P.d_epoch.mean()),
                   d_test_close=float(close.d_test.mean()),
                   d_test_far=float(far.d_test.mean()), n_pairs=int(len(P))),
              open(RES / "noise_floor.json", "w"), indent=2)

    print()
    print("=" * 84)
    print("TABLE 5  Stratified in-domain performance")
    print("=" * 84)
    rows = []
    for m in MODELS:
        line = []
        for t in ["small", "medium", "large"]:
            s = kv[(kv.model == m) & (kv.size_tertile == t)]
            line.append(s.dice.mean())
            rows.append(dict(model=PRETTY[m], stratum=t, n=len(s), dice=s.dice.mean()))
        sm = kv[(kv.model == m) & (kv.size_tertile == "small")].dice.values
        lg = kv[(kv.model == m) & (kv.size_tertile == "large")].dice.values
        d, lo, hi = boot_diff(sm, lg)
        p = perm_test(sm, lg)
        print(f"  {PRETTY[m]:<14} small {line[0]:.4f}  medium {line[1]:.4f}  "
              f"large {line[2]:.4f}   small-large {d:+.4f} "
              f"[{lo:+.3f},{hi:+.3f}] {stars(p)}")
    print()
    for m in MODELS:
        ses = kv[(kv.model == m) & kv.sessile].dice.values
        non = kv[(kv.model == m) & ~kv.sessile].dice.values
        d, lo, hi = boot_diff(ses, non)
        p = perm_test(ses, non)
        rows.append(dict(model=PRETTY[m], stratum="sessile", n=len(ses), dice=ses.mean()))
        rows.append(dict(model=PRETTY[m], stratum="non-sessile", n=len(non), dice=non.mean()))
        print(f"  {PRETTY[m]:<14} sessile {ses.mean():.4f}  non-sessile {non.mean():.4f}  "
              f"gap {d:+.4f} [{lo:+.3f},{hi:+.3f}] {stars(p)}")
    pd.DataFrame(rows).to_csv(RES / "t5_strata_indomain.csv", index=False)

    print()
    print("=" * 84)
    print("TABLE 6  Out-of-domain size strata (external images, true small lesions)")
    print("=" * 84)
    ext = ext.copy()
    ext["size_bin"] = pd.cut(ext.area_frac_gt if "area_frac_gt" in ext else ext.area_frac,
                             bins=[0, 0.01, 0.05, 1.0],
                             labels=["<1%", "1-5%", ">5%"])
    rows = []
    for m in MODELS:
        line = []
        for b in ["<1%", "1-5%", ">5%"]:
            s = ext[(ext.model == m) & (ext.size_bin == b)]
            line.append((s.dice.mean(), s.det_rate_iou05.mean(), len(s)))
            rows.append(dict(model=PRETTY[m], size_bin=b, n=len(s),
                             dice=s.dice.mean(), det=s.det_rate_iou05.mean()))
        print(f"  {PRETTY[m]:<14}" + "  ".join(
            f"{b} Dice {v[0]:.4f} det {v[1]:.3f}"
            for b, v in zip(["<1%", "1-5%", ">5%"], line)))
    pd.DataFrame(rows).to_csv(RES / "t6_strata_external.csv", index=False)
    # Each external image is scored by all five fold models, so row counts are
    # 5x the number of distinct images; report distinct images.
    # Filenames repeat across external datasets ("1", "2", ...), so identity
    # is the (dataset, name) pair rather than the name alone.
    n_small = len(ext[(ext.model == MODELS[0]) & (ext.size_bin == "<1%")]
                  .drop_duplicates(subset=["dataset", "name"]))
    n_small_kv = int((pd.read_csv(OUT / "metadata.csv").area_frac < 0.01).sum())
    print(f"  ({n_small} distinct external images have lesion area <1%; "
          f"Kvasir-SEG has {n_small_kv} of 1000)")

    print()
    print("=" * 84)
    print("TABLE 7  Test-time augmentation")
    print("=" * 84)
    rows = []
    for m in MODELS:
        for ds in DATASETS:
            b = cv[(cv.model == m) & (cv.dataset == ds)].dice.mean()
            t = tta[(tta.model == m) & (tta.dataset == ds)].dice.mean()
            rows.append(dict(model=PRETTY[m], dataset=DSNAME[ds], base=b,
                             tta=t, delta=t - b))
    T = pd.DataFrame(rows)
    T.to_csv(RES / "t7_tta.csv", index=False)
    piv = T.pivot(index="model", columns="dataset", values="delta")
    print(piv.round(4).to_string())
    print(f"\n  mean gain in-domain {T[T.dataset == 'Kvasir-SEG'].delta.mean():+.4f}  "
          f"vs external {T[T.dataset != 'Kvasir-SEG'].delta.mean():+.4f}")

    print(f"\nWrote CSVs to {RES}")


if __name__ == "__main__":
    main()
