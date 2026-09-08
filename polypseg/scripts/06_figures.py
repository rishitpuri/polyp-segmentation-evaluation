"""Generate the paper figures from the saved per-image metrics.

Figures (PDF for LaTeX + PNG for quick viewing):
  fig1_split_vs_cv     single-split vs 5-fold CV: the spurious finding
  fig2_dice_vs_area    Dice against polyp size, with binned trend
  fig3_strata_box      distribution per stratum and architecture
  fig4_cross_dataset   in-domain vs CVC-ClinicDB, per fold
  fig5_efficiency      parameters vs Dice vs fold-to-fold stability
  fig6_tta             effect of test-time augmentation
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "unetpp", "segformer"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0"}
COLORS = {"unet": "#4C72B0", "unetpp": "#DD8452", "segformer": "#55A868"}
PARAMS = {"unet": 24.44, "unetpp": 26.08, "segformer": 3.71}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def boot_ci(v, n_boot=5000, seed=0):
    v = np.asarray([x for x in v if not np.isnan(x)])
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, len(v)), replace=True).mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load():
    cv = pd.read_csv(OUT / "analysis_cv" / "cv_perimage_merged.csv")
    cv["sessile"] = cv["sessile"].astype(bool)
    cv_tta = pd.read_csv(OUT / "analysis_cv" / "cv_perimage_merged_tta.csv")
    cv_tta["sessile"] = cv_tta["sessile"].astype(bool)

    # single-split results (3 seeds, 100-image test set)
    single = pd.read_csv(OUT / "analysis" / "all_perimage_merged.csv")
    single["sessile"] = single["sessile"].astype(bool)
    single = single[~single.tta]
    return cv, cv_tta, single


def fig1_split_vs_cv(cv, single):
    """The methodological centrepiece: the sessile penalty measured on a
    single 100-image split vs under 5-fold CV over all 1000 images."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (df, title, n_ses) in zip(axes, [
        (single[single.dataset == "kvasir-seg"], "Single 80/10/10 split", 19),
        (cv[cv.dataset == "kvasir-seg"], "5-fold cross-validation", 196),
    ]):
        x = np.arange(len(MODELS))
        for i, m in enumerate(MODELS):
            for j, (flag, hatch, lbl) in enumerate(
                    [(False, "", "non-sessile"), (True, "//", "sessile (flat)")]):
                sub = df[(df.model == m) & (df.sessile == flag)].dice
                mean, lo, hi = boot_ci(sub.values)
                ax.bar(i + (j - 0.5) * 0.36, mean, 0.34, color=COLORS[m],
                       alpha=1.0 if not flag else 0.55, hatch=hatch,
                       edgecolor="white",
                       yerr=[[mean - lo], [hi - mean]], capsize=3,
                       label=lbl if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[m] for m in MODELS], fontsize=8)
        ax.set_title(f"{title}\n(n={n_ses} sessile test images)", fontsize=9)
        ax.set_ylim(0.70, 0.98)
    axes[0].set_ylabel("Dice")
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[0].annotate("apparent immunity\n(false positive)",
                     xy=(1.18, 0.914), xytext=(1.0, 0.74), fontsize=7,
                     ha="center", arrowprops=dict(arrowstyle="->", lw=0.8))
    save(fig, "fig1_split_vs_cv")


def fig2_dice_vs_area(cv):
    kv = cv[cv.dataset == "kvasir-seg"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m in MODELS:
        s = kv[kv.model == m]
        ax.scatter(s.area_frac * 100, s.dice, s=4, alpha=0.12, color=COLORS[m])
        bins = np.percentile(s.area_frac * 100, np.linspace(0, 100, 11))
        centres, means = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = s[(s.area_frac * 100 >= a) & (s.area_frac * 100 < b)]
            if len(sel):
                centres.append(np.sqrt(a * b) if a > 0 else b / 2)
                means.append(sel.dice.mean())
        ax.plot(centres, means, "-o", ms=4, lw=1.8, color=COLORS[m], label=PRETTY[m])
    ax.set_xscale("log")
    ax.set_xlabel("Polyp size (% of image area, log scale)")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Performance is non-monotonic in polyp size", fontsize=9)
    save(fig, "fig2_dice_vs_area")


def fig3_strata_box(cv):
    kv = cv[cv.dataset == "kvasir-seg"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    strata = [("size_tertile", ["small", "medium", "large"], "Polyp size tertile"),
              ("sessile", [False, True], "Morphology")]
    for ax, (col, levels, xlabel) in zip(axes, strata):
        positions, data, colors_ = [], [], []
        for li, lv in enumerate(levels):
            for mi, m in enumerate(MODELS):
                positions.append(li * (len(MODELS) + 1) + mi)
                data.append(kv[(kv.model == m) & (kv[col] == lv)].dice.values)
                colors_.append(COLORS[m])
        # Outliers are shown deliberately: the sessile penalty is carried by a
        # tail of catastrophic misses, which hiding fliers would conceal.
        bp = ax.boxplot(data, positions=positions, widths=0.75, showfliers=True,
                        flierprops=dict(marker="o", ms=2, alpha=0.35,
                                        markerfacecolor="0.3", markeredgecolor="none"),
                        patch_artist=True, medianprops=dict(color="black", lw=1.2))
        for patch, c in zip(bp["boxes"], colors_):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks([li * (len(MODELS) + 1) + 1 for li in range(len(levels))])
        ax.set_xticklabels(["non-sessile", "sessile (flat)"] if col == "sessile"
                           else [str(l) for l in levels], fontsize=8)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Dice")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], alpha=0.75) for m in MODELS]
    axes[1].legend(handles, [PRETTY[m] for m in MODELS], fontsize=7.5, loc="lower left")
    save(fig, "fig3_strata_box")


def fig4_cross_dataset(cv):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, m in enumerate(MODELS):
        ind = cv[(cv.model == m) & (cv.dataset == "kvasir-seg")].groupby("fold").dice.mean()
        ood = cv[(cv.model == m) & (cv.dataset == "cvc-clinicdb")].groupby("fold").dice.mean()
        for f in ind.index:
            ax.plot([i - 0.16, i + 0.16], [ind[f], ood[f]], "-", color=COLORS[m],
                    alpha=0.45, lw=1)
        ax.plot([i - 0.16] * len(ind), ind.values, "o", color=COLORS[m], ms=5)
        ax.plot([i + 0.16] * len(ood), ood.values, "s", color=COLORS[m], ms=5,
                markerfacecolor="white")
        ax.plot([i - 0.16, i + 0.16], [ind.mean(), ood.mean()], "-",
                color="black", lw=2, zorder=5)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([PRETTY[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("Dice")
    ax.set_title("Kvasir-SEG (circles) $\\rightarrow$ CVC-ClinicDB (squares)\n"
                 "each line is one cross-validation fold", fontsize=9)
    save(fig, "fig4_cross_dataset")


def fig5_efficiency(cv):
    kv = cv[cv.dataset == "kvasir-seg"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for m in MODELS:
        s = kv[kv.model == m]
        per_fold = s.groupby("fold").dice.mean()
        axes[0].scatter(PARAMS[m], s.dice.mean(), s=90, color=COLORS[m], label=PRETTY[m])
        axes[0].annotate(PRETTY[m], (PARAMS[m], s.dice.mean()),
                         textcoords="offset points", xytext=(6, -10), fontsize=7.5)
        axes[1].scatter(PARAMS[m], per_fold.std(), s=90, color=COLORS[m])
        axes[1].annotate(PRETTY[m], (PARAMS[m], per_fold.std()),
                         textcoords="offset points", xytext=(6, -10), fontsize=7.5)
    axes[0].set_xlabel("Parameters (M)")
    axes[0].set_ylabel("Dice (mean over 1000 images)")
    axes[0].set_xlim(0, 32)
    axes[1].set_xlabel("Parameters (M)")
    axes[1].set_ylabel("Fold-to-fold std of Dice")
    axes[1].set_xlim(0, 32)
    axes[1].set_title("Lower is more reproducible", fontsize=9)
    save(fig, "fig5_efficiency")


def fig6_tta(cv, cv_tta):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    width = 0.35
    x = np.arange(len(MODELS))
    for j, ds in enumerate(["kvasir-seg", "cvc-clinicdb"]):
        deltas = []
        for m in MODELS:
            b = cv[(cv.model == m) & (cv.dataset == ds)].dice.mean()
            t = cv_tta[(cv_tta.model == m) & (cv_tta.dataset == ds)].dice.mean()
            deltas.append(t - b)
        ax.bar(x + (j - 0.5) * width, deltas, width,
               label="in-domain" if ds == "kvasir-seg" else "cross-dataset",
               color="#4C72B0" if j == 0 else "#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("$\\Delta$ Dice from flip TTA")
    ax.legend(fontsize=7.5)
    ax.set_title("TTA helps roughly twice as much out-of-domain", fontsize=9)
    save(fig, "fig6_tta")


def main():
    cv, cv_tta, single = load()
    print("Generating figures:")
    fig1_split_vs_cv(cv, single)
    fig2_dice_vs_area(cv)
    fig3_strata_box(cv)
    fig4_cross_dataset(cv)
    fig5_efficiency(cv)
    fig6_tta(cv, cv_tta)
    print(f"\nAll figures in {FIG}")


if __name__ == "__main__":
    main()
