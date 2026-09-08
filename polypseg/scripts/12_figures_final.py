"""Figures for the journal manuscript.

  f1_small_polyp     the headline: detection collapses on clinically small
                     lesions, which Kvasir-SEG almost entirely lacks
  f2_resolving       architecture spread against the measured noise floor
  f3_selection       run-to-run variance and its cause (checkpoint selection)
  f4_split_vs_cv     the single-split finding that did not replicate
  f5_cross_dataset   degradation across all five test sets
  f6_strata          per-image Dice distributions by stratum
  f7_tta             test-time augmentation, in-domain vs out-of-domain
"""
import glob
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
FIG = OUT / "figures_final"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "unetpp", "segformer", "segformer_b2", "unet_pvt"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0",
          "segformer_b2": "SegFormer-B2", "unet_pvt": "U-Net+PVTv2"}
COLORS = {"unet": "#4C72B0", "unetpp": "#DD8452", "segformer": "#55A868",
          "segformer_b2": "#C44E52", "unet_pvt": "#8172B3"}
DATASETS = ["kvasir-seg", "cvc-clinicdb", "cvc-300", "cvc-colondb", "etis-larib"]
DSNAME = {"kvasir-seg": "Kvasir-SEG\n(in-domain)", "cvc-clinicdb": "CVC-ClinicDB",
          "cvc-300": "CVC-300", "cvc-colondb": "CVC-ColonDB",
          "etis-larib": "ETIS-Larib"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def load_cv(tta=False):
    fr = []
    for m in MODELS:
        for k in range(5):
            p = OUT / "cv2" / f"perimage_{m}_seed42_fold{k}{'_tta' if tta else ''}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d["fold"] = k
                fr.append(d)
    return pd.concat(fr, ignore_index=True)


def load_repeats():
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
            recs.append(dict(model=s["model"], cond=cond, epoch=s["best_epoch"],
                             val=s["best_val_dice"], test=d.dice.mean()))
    return pd.DataFrame(recs)


def prep():
    cv = load_cv()
    ext_meta = pd.read_csv(OUT / "external_metadata.csv")[
        ["dataset", "name", "area_frac", "n_lesions"]]
    ext_meta["name"] = ext_meta["name"].astype(str)
    kv_meta = pd.read_csv(OUT / "cv_assignment.csv")[
        ["name", "size_tertile", "area_frac", "sessile"]]

    kv = cv[cv.dataset == "kvasir-seg"].merge(kv_meta, on="name", how="left")
    kv["sessile"] = kv["sessile"].astype(bool)

    e = cv[cv.dataset != "kvasir-seg"].copy()
    e["name"] = e["name"].astype(str)
    e = e.merge(ext_meta, on=["dataset", "name"], how="left")
    return cv, kv, e, kv_meta, ext_meta


def f1_small_polyp(kv, ext, kv_meta, ext_meta):
    """Headline figure: performance against true lesion size, plus the
    size distribution showing Kvasir-SEG contains almost no small lesions."""
    bins = [0, 0.01, 0.05, 1.0]
    labels = ["<1%", "1-5%", ">5%"]
    kv = kv.copy()
    kv["size_bin"] = pd.cut(kv.area_frac, bins=bins, labels=labels)
    ext = ext.copy()
    ext["size_bin"] = pd.cut(ext.area_frac, bins=bins, labels=labels)
    allimg = pd.concat([kv.assign(src="Kvasir-SEG"), ext.assign(src="external")],
                       ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    x = np.arange(len(labels))
    w = 0.15
    for i, m in enumerate(MODELS):
        dice = [allimg[(allimg.model == m) & (allimg.size_bin == b)].dice.mean()
                for b in labels]
        det = [allimg[(allimg.model == m) & (allimg.size_bin == b)].det_rate_iou05.mean()
               for b in labels]
        axes[0].bar(x + (i - 2) * w, dice, w, color=COLORS[m], label=PRETTY[m])
        axes[1].bar(x + (i - 2) * w, det, w, color=COLORS[m])
    for ax, ylab, title in [(axes[0], "Dice", "Segmentation quality"),
                            (axes[1], "Per-lesion detection rate @IoU 0.5",
                             "Detection")]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Lesion size (% of image area)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.0)
    axes[1].axhline(0.5, ls=":", c="0.4", lw=1)
    axes[1].text(0.02, 0.52, "half of lesions missed", fontsize=7, color="0.3")
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")

    # size distribution per dataset
    counts = []
    for ds in ["kvasir-seg", "cvc-clinicdb", "cvc-300", "cvc-colondb", "etis-larib"]:
        if ds == "kvasir-seg":
            a = kv_meta.area_frac
        else:
            a = ext_meta[ext_meta.dataset == ds].area_frac
        counts.append(100 * (a < 0.01).mean())
    names = ["Kvasir\nSEG", "CVC\nClinicDB", "CVC\n300", "CVC\nColonDB", "ETIS\nLarib"]
    bars = axes[2].bar(names, counts, color=["#B00000"] + ["#7f7f7f"] * 4)
    axes[2].set_ylabel("% of images with\nlesion <1% of area")
    axes[2].set_title("Where small lesions actually occur", fontsize=9)
    for b, c in zip(bars, counts):
        axes[2].text(b.get_x() + b.get_width() / 2, c + 0.7, f"{c:.1f}",
                     ha="center", fontsize=7.5)
    axes[2].tick_params(axis="x", labelsize=7)
    save(fig, "f1_small_polyp")


def f2_resolving(cv, rep):
    sigma = rep[rep.cond.isin(["A", "C", "D"])].groupby(["model", "cond"]).test.std().mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    spreads, ratios = [], []
    for ds in DATASETS:
        means = [cv[(cv.model == m) & (cv.dataset == ds)]
                 .groupby("fold").dice.mean().mean() for m in MODELS]
        spreads.append(max(means) - min(means))
        ratios.append((max(means) - min(means)) / sigma)
    xs = np.arange(len(DATASETS))
    ax.bar(xs, ratios, color=["#4C72B0"] + ["#C44E52"] * 4, alpha=0.85)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.text(len(DATASETS) - 0.45, 1.6, "noise floor ($\\sigma_{run}$)", fontsize=7.5,
            ha="right")
    ax.axhline(2, color="0.5", ls=":", lw=1)
    for i, (r, s) in enumerate(zip(ratios, spreads)):
        ax.text(i, r + 0.5, f"{r:.1f}$\\sigma$\n({s:.3f})", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([DSNAME[d].replace("\n", " ") for d in DATASETS],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Architecture spread / $\\sigma_{run}$")
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.set_title("Cross-dataset evaluation resolves architectures that\n"
                 "in-domain evaluation cannot", fontsize=9)
    save(fig, "f2_resolving")


def f3_selection(rep):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    # left: spread per condition
    labels = {"A": "default\nRNG+kernels", "B": "determ.\nkernels",
              "C": "fixed RNG", "D": "fixed RNG +\ndeterm."}
    conds = [c for c in ["A", "B", "C", "D"] if (rep.cond == c).any()]
    for i, c in enumerate(conds):
        g = rep[rep.cond == c]
        sp = g.groupby("model").test.agg(lambda s: s.max() - s.min())
        axes[0].scatter([i] * len(sp), sp.values, s=45,
                        color=[COLORS[m] for m in sp.index], zorder=3)
        axes[0].hlines(sp.mean(), i - 0.25, i + 0.25, color="k", lw=2)
    axes[0].set_xticks(range(len(conds)))
    axes[0].set_xticklabels([labels[c] for c in conds], fontsize=7.5)
    axes[0].set_ylabel("Test Dice spread across\nidentical repeated runs")
    axes[0].set_ylim(0, None)
    axes[0].set_title("Determinism settings do not remove\nrun-to-run variance",
                      fontsize=9)

    # right: mechanism
    pairs = []
    for (m, c), g in rep.groupby(["model", "cond"]):
        for a, b in itertools.combinations(g.to_dict("records"), 2):
            pairs.append((abs(a["epoch"] - b["epoch"]), abs(a["test"] - b["test"]), m))
    P = pd.DataFrame(pairs, columns=["d_epoch", "d_test", "model"])
    for m in MODELS:
        s = P[P.model == m]
        axes[1].scatter(s.d_epoch, s.d_test, s=28, color=COLORS[m],
                        alpha=0.8, label=PRETTY[m])
    if len(P) > 2:
        z = np.polyfit(P.d_epoch, P.d_test, 1)
        xr = np.linspace(0, P.d_epoch.max(), 20)
        axes[1].plot(xr, np.polyval(z, xr), "k--", lw=1.2,
                     label=f"r = {P.d_epoch.corr(P.d_test):.2f}")
    axes[1].set_xlabel("| difference in selected epoch |")
    axes[1].set_ylabel("| difference in test Dice |")
    axes[1].set_title("Cause: checkpoint selection on a\nflat validation curve", fontsize=9)
    axes[1].legend(fontsize=6.5, ncol=2)
    save(fig, "f3_selection")


def f4_split_vs_cv(kv):
    single = pd.read_csv(OUT / "analysis" / "all_perimage_merged.csv")
    single = single[(~single.tta) & (single.dataset == "kvasir-seg")].copy()
    single["sessile"] = single["sessile"].astype(bool)
    three = ["unet", "unetpp", "segformer"]

    def ci(v, seed=0):
        rng = np.random.default_rng(seed)
        v = np.asarray(v)
        b = rng.choice(v, (5000, len(v)), replace=True).mean(1)
        return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, (df, title, n) in zip(axes, [
            (single, "Single 80/10/10 split", 19),
            (kv[kv.model.isin(three)], "5-fold cross-validation", 196)]):
        for i, m in enumerate(three):
            for j, (flag, hatch, lab) in enumerate([(False, "", "non-sessile"),
                                                    (True, "//", "sessile (flat)")]):
                v = df[(df.model == m) & (df.sessile == flag)].dice.values
                mean, lo, hi = ci(v)
                ax.bar(i + (j - 0.5) * 0.36, mean, 0.34, color=COLORS[m],
                       alpha=1.0 if not flag else 0.55, hatch=hatch,
                       edgecolor="white", yerr=[[mean - lo], [hi - mean]], capsize=3,
                       label=lab if i == 0 else None)
        ax.set_xticks(range(len(three)))
        ax.set_xticklabels([PRETTY[m] for m in three], fontsize=8)
        ax.set_title(f"{title}\n({n} sessile test images)", fontsize=9)
        ax.set_ylim(0.70, 0.98)
    axes[0].set_ylabel("Dice")
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[0].annotate("apparent immunity\n(did not replicate)", xy=(1.18, 0.914),
                     xytext=(0.95, 0.735), fontsize=7, ha="center",
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    save(fig, "f4_split_vs_cv")


def f5_cross_dataset(cv):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    xs = np.arange(len(DATASETS))
    for m in MODELS:
        ys = [cv[(cv.model == m) & (cv.dataset == d)].groupby("fold").dice.mean().mean()
              for d in DATASETS]
        es = [cv[(cv.model == m) & (cv.dataset == d)].groupby("fold").dice.mean().std()
              for d in DATASETS]
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.6, capsize=3,
                    color=COLORS[m], label=PRETTY[m])
    ax.set_xticks(xs)
    ax.set_xticklabels([DSNAME[d].replace("\n", " ") for d in DATASETS],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Dice")
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("Zero-shot transfer; error bars are sd over 5 fold-models", fontsize=9)
    save(fig, "f5_cross_dataset")


def f6_strata(kv):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), sharey=True)
    for ax, (col, levels, xlab) in zip(axes, [
            ("size_tertile", ["small", "medium", "large"], "Polyp size tertile (in-domain)"),
            ("sessile", [False, True], "Morphology")]):
        pos, data, cols = [], [], []
        for li, lv in enumerate(levels):
            for mi, m in enumerate(MODELS):
                pos.append(li * (len(MODELS) + 1) + mi)
                data.append(kv[(kv.model == m) & (kv[col] == lv)].dice.values)
                cols.append(COLORS[m])
        bp = ax.boxplot(data, positions=pos, widths=0.78, showfliers=True,
                        flierprops=dict(marker="o", ms=1.8, alpha=0.3,
                                        markerfacecolor="0.3", markeredgecolor="none"),
                        patch_artist=True, medianprops=dict(color="black", lw=1.1))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks([li * (len(MODELS) + 1) + 2 for li in range(len(levels))])
        ax.set_xticklabels(["non-sessile", "sessile (flat)"] if col == "sessile"
                           else levels, fontsize=8)
        ax.set_xlabel(xlab)
    axes[0].set_ylabel("Dice")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], alpha=0.75) for m in MODELS]
    axes[1].legend(handles, [PRETTY[m] for m in MODELS], fontsize=7, loc="lower left")
    save(fig, "f6_strata")


def f7_tta(cv, tta):
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    xs = np.arange(len(DATASETS))
    w = 0.15
    for i, m in enumerate(MODELS):
        d = [tta[(tta.model == m) & (tta.dataset == ds)].dice.mean()
             - cv[(cv.model == m) & (cv.dataset == ds)].dice.mean() for ds in DATASETS]
        ax.bar(xs + (i - 2) * w, d, w, color=COLORS[m], label=PRETTY[m])
    ax.set_xticks(xs)
    ax.set_xticklabels([DSNAME[d].replace("\n", " ") for d in DATASETS],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("$\\Delta$ Dice from flip TTA")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("TTA gains are larger under domain shift", fontsize=9)
    save(fig, "f7_tta")


def main():
    cv, kv, ext, kv_meta, ext_meta = prep()
    tta = load_cv(tta=True)
    rep = load_repeats()
    print("Generating figures:")
    f1_small_polyp(kv, ext, kv_meta, ext_meta)
    f2_resolving(cv, rep)
    f3_selection(rep)
    f4_split_vs_cv(kv)
    f5_cross_dataset(cv)
    f6_strata(kv)
    f7_tta(cv, tta)
    print(f"\nAll figures in {FIG}")


if __name__ == "__main__":
    main()
