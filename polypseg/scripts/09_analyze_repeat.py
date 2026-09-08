"""Quantify run-to-run variance and compare it against the differences
between architectures.

The question: identical command, identical seed, same data, same fold --
how much does test Dice move? Whatever spread remains is implementation
non-determinism (cuDNN kernel selection, atomic accumulation order), and it
sets the noise floor below which architecture comparisons are meaningless.

Phase A runs use default kernels; Phase B runs add --deterministic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
REP = OUT / "repeat"
CV = OUT / "cv2"
RESULTS = OUT / "analysis_repeat"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "unetpp", "segformer", "segformer_b2", "unet_pvt"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0",
          "segformer_b2": "SegFormer-B2", "unet_pvt": "U-Net+PVTv2"}
DATASETS = ["kvasir-seg", "cvc-clinicdb", "cvc-300", "cvc-colondb", "etis-larib"]


def load(pattern, tags):
    rows = []
    for m in MODELS:
        for t in tags:
            p = REP / f"perimage_{m}_seed42_{pattern.format(t=t)}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d["model"] = m
            d["rep"] = t
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    nondet = load("fold0rep{t}", [1, 2, 3])
    det = load("fold0det{t}", [1, 2])
    if nondet.empty:
        raise SystemExit("no repeat results found")

    print("=" * 78)
    print("TABLE A  Repeated identical runs, default kernels (fold 0 test, n=200)")
    print("=" * 78)
    rows = []
    for m in MODELS:
        s = nondet[(nondet.model == m) & (nondet.dataset == "kvasir-seg")]
        if s.empty:
            continue
        per_rep = s.groupby("rep").dice.mean()
        rows.append(dict(model=PRETTY[m], n_runs=len(per_rep), mean=per_rep.mean(),
                         sd=per_rep.std(), spread=per_rep.max() - per_rep.min(),
                         runs=list(per_rep.round(4))))
        print(f"  {PRETTY[m]:<14} runs={list(per_rep.round(4))}  "
              f"sd={per_rep.std():.4f}  spread={per_rep.max() - per_rep.min():.4f}")
    tab_a = pd.DataFrame(rows)
    tab_a.to_csv(RESULTS / "repeat_nondeterministic.csv", index=False)

    sd_run = tab_a.sd.mean()
    spread_run = tab_a.spread.max()
    print(f"\n  mean within-config sd  sigma_run = {sd_run:.4f}")
    print(f"  worst within-config spread        = {spread_run:.4f}")

    if not det.empty:
        print()
        print("=" * 78)
        print("TABLE B  Repeated identical runs with --deterministic")
        print("=" * 78)
        rows = []
        for m in MODELS:
            s = det[(det.model == m) & (det.dataset == "kvasir-seg")]
            if s.empty:
                continue
            per_rep = s.groupby("rep").dice.mean()
            rows.append(dict(model=PRETTY[m], n_runs=len(per_rep),
                             spread=per_rep.max() - per_rep.min(),
                             runs=list(per_rep.round(6))))
            print(f"  {PRETTY[m]:<14} runs={list(per_rep.round(6))}  "
                  f"spread={per_rep.max() - per_rep.min():.6f}")
        pd.DataFrame(rows).to_csv(RESULTS / "repeat_deterministic.csv", index=False)

    # Architecture differences from the 5-fold CV, expressed in units of sigma_run
    print()
    print("=" * 78)
    print("TABLE C  Architecture gaps vs the noise floor (in-domain and OOD)")
    print("=" * 78)
    cv_rows = []
    for m in MODELS:
        for k in range(5):
            p = CV / f"perimage_{m}_seed42_fold{k}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d["fold"] = k
                cv_rows.append(d)
    cv = pd.concat(cv_rows, ignore_index=True)

    summary = []
    for ds in DATASETS:
        means = {m: cv[(cv.model == m) & (cv.dataset == ds)]
                 .groupby("fold").dice.mean().mean() for m in MODELS}
        best, worst = max(means, key=means.get), min(means, key=means.get)
        spread = means[best] - means[worst]
        summary.append(dict(dataset=ds, best=PRETTY[best], worst=PRETTY[worst],
                            best_dice=means[best], worst_dice=means[worst],
                            spread=spread, spread_in_sigma=spread / sd_run))
        print(f"  {ds:<14} spread={spread:.4f} ({spread / sd_run:5.1f} x sigma_run)  "
              f"best={PRETTY[best]} {means[best]:.4f}  worst={PRETTY[worst]} {means[worst]:.4f}")
    pd.DataFrame(summary).to_csv(RESULTS / "arch_gap_vs_noise.csv", index=False)

    print()
    print("=" * 78)
    print("TABLE D  Which pairwise in-domain gaps exceed the noise floor?")
    print("=" * 78)
    kv_means = {m: cv[(cv.model == m) & (cv.dataset == "kvasir-seg")]
                .groupby("fold").dice.mean().mean() for m in MODELS}
    pairs = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            gap = abs(kv_means[a] - kv_means[b])
            ratio = gap / sd_run
            verdict = "resolvable" if ratio >= 2 else "BELOW NOISE"
            pairs.append(dict(a=PRETTY[a], b=PRETTY[b], gap=gap,
                              ratio=ratio, verdict=verdict))
            print(f"  {PRETTY[a]:<14} vs {PRETTY[b]:<14} gap={gap:.4f}  "
                  f"{ratio:4.1f}x sigma_run   {verdict}")
    pd.DataFrame(pairs).to_csv(RESULTS / "pairwise_vs_noise.csv", index=False)

    (RESULTS / "sigma_run.txt").write_text(f"{sd_run:.6f}\n")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
