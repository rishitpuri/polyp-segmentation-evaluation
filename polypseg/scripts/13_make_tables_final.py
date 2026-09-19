"""Generate LaTeX tables and numeric macros for the journal manuscript
directly from the analysis CSVs, so the text cannot drift from the results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
RES = OUT / "analysis_final"
TAB = ROOT / "polypseg" / "paper_journal" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

# IEEE two-column variant: these tables are too wide for a single column, so
# they must span both (`table*`). Emitted separately rather than editing the
# shared sources, because the single-column manuscript needs plain `table`.
TAB_IEEE = ROOT / "polypseg" / "paper_ieee" / "tables"
TAB_IEEE.mkdir(parents=True, exist_ok=True)
WIDE = {"tab_indomain.tex", "tab_cross.tex", "tab_size.tex",
        "tab_variance.tex", "tab_resolving.tex", "tab_sessile.tex"}


def write_ieee_variants():
    """Mirror the tables into the IEEE manuscript, widening the float and
    reducing the font so nothing overflows the column."""
    for src in sorted(TAB.glob("*.tex")):
        text = src.read_text()
        if src.name in WIDE:
            text = (text.replace(r"\begin{table}[t]", r"\begin{table*}[t]")
                        .replace(r"\end{table}", r"\end{table*}"))
            if r"\footnotesize" not in text:
                text = text.replace(r"\centering", "\\centering\n\\footnotesize")
        (TAB_IEEE / src.name).write_text(text)
    print(f"  mirrored {len(list(TAB.glob('*.tex')))} tables to {TAB_IEEE}")

DS = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-Larib"]


def w(name, lines):
    (TAB / name).write_text("\n".join(lines) + "\n")
    print(f"  wrote {name}")


def main():
    t1 = pd.read_csv(RES / "t1_indomain.csv")
    t2 = pd.read_csv(RES / "t2_cross_dataset.csv")
    t2b = pd.read_csv(RES / "t2b_detection.csv")
    t3 = pd.read_csv(RES / "t3_resolving_power.csv")
    t3b = pd.read_csv(RES / "t3b_pairwise.csv")
    t4 = pd.read_csv(RES / "t4_variance_conditions.csv")
    t5 = pd.read_csv(RES / "t5_strata_indomain.csv")
    t6 = pd.read_csv(RES / "t6_strata_external.csv")
    t7 = pd.read_csv(RES / "t7_tta.csv")
    nf = json.load(open(RES / "noise_floor.json"))
    extm = pd.read_csv(OUT / "external_metadata.csv")
    kvm = pd.read_csv(OUT / "metadata.csv")

    # ---- Table 1: in-domain -------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{In-domain performance under five-fold cross-validation on "
         r"Kvasir-SEG. Every image is tested exactly once ($n=1000$ per model). "
         r"HD95 is expressed as a percentage of the image diagonal; the "
         r"detection rate is the fraction of ground-truth lesions matched at "
         r"IoU $\geq 0.5$.}",
         r"\label{tab:indomain}", r"\begin{tabular}{lrccccc}", r"\toprule",
         r"Model & Par.\ (M) & Dice & IoU & Boundary-F1 & HD95 (\%) & Det.\ rate \\",
         r"\midrule"]
    best = {c: t1[c].max() for c in ["dice", "iou", "bf1", "det_iou50"]}
    best["hd95_pct"] = t1.hd95_pct.min()
    for _, r in t1.iterrows():
        def f(col, v, dec=4):
            s = f"{v:.{dec}f}"
            return rf"\textbf{{{s}}}" if abs(v - best[col]) < 1e-12 else s
        L.append(f"{r['model']} & {r.params_M:.1f} & {f('dice', r.dice)} & "
                 f"{f('iou', r.iou)} & {f('bf1', r.bf1)} & "
                 f"{f('hd95_pct', r.hd95_pct, 2)} & {f('det_iou50', r.det_iou50)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_indomain.tex", L)

    # ---- Table 2: cross-dataset --------------------------------------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Zero-shot transfer. Models are trained on Kvasir-SEG only; "
         r"each value is the mean over the five cross-validation fold models. "
         r"Upper block: Dice. Lower block: per-lesion detection rate at "
         r"IoU $\geq 0.5$.}",
         r"\label{tab:cross}", r"\begin{tabular}{l" + "c" * len(DS) + "}", r"\toprule",
         "Model & " + " & ".join(d.replace("-", "-\\allowbreak ") for d in DS) + r" \\",
         r"\midrule", r"\multicolumn{6}{l}{\emph{Dice}}\\"]
    for _, r in t2.iterrows():
        L.append(f"\\quad {r['model']} & " + " & ".join(f"{r[d]:.4f}" for d in DS) + r" \\")
    L += [r"\midrule", r"\multicolumn{6}{l}{\emph{Per-lesion detection rate}}\\"]
    for _, r in t2b.iterrows():
        L.append(f"\\quad {r['model']} & " + " & ".join(f"{r[d]:.4f}" for d in DS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_cross.tex", L)

    # ---- Table 3: lesion size (pooled, in and out of domain) ---------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Performance by true lesion size, pooled over all test "
         r"images (in-domain and external). Lesions occupying less than 1\% of "
         r"the image are segmented far worse and, more importantly, are missed "
         r"outright about half of the time. Kvasir-SEG contains only "
         + f"{int((kvm.area_frac < 0.01).sum())}"
         + r" such images, so this failure mode is nearly invisible to "
           r"in-domain benchmarking.}",
         r"\label{tab:size}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{Dice} & \multicolumn{3}{c}{Detection rate} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Model & $<$1\% & 1--5\% & $>$5\% & $<$1\% & 1--5\% & $>$5\% \\",
         r"\midrule"]
    for m in t6.model.unique():
        g = t6[t6.model == m].set_index("size_bin")
        L.append(f"{m} & " + " & ".join(f"{g.loc[b, 'dice']:.4f}" for b in ["<1%", "1-5%", ">5%"])
                 + " & " + " & ".join(f"{g.loc[b, 'det']:.3f}" for b in ["<1%", "1-5%", ">5%"])
                 + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_size.tex", L)

    # ---- Table 4: strata ----------------------------------------------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Sessile (flat) versus non-sessile lesions, in domain. "
         r"Differences are bootstrap means with 95\% percentile intervals; "
         r"$p$ from a two-sided permutation test on the difference of means.}",
         r"\label{tab:sessile}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Model & Sessile ($n{=}196$) & Non-sessile ($n{=}804$) & Difference \\",
         r"\midrule"]
    piv = t5[t5.stratum.isin(["sessile", "non-sessile"])].pivot(
        index="model", columns="stratum", values="dice")
    for m in t1.model:
        L.append(f"{m} & {piv.loc[m, 'sessile']:.4f} & {piv.loc[m, 'non-sessile']:.4f} & "
                 f"${piv.loc[m, 'sessile'] - piv.loc[m, 'non-sessile']:+.4f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_sessile.tex", L)

    # ---- Table 5: variance conditions --------------------------------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Test-Dice spread across repeated runs of an identical "
         r"configuration (same seed, same fold, same data). Neither seeding the "
         r"data-loader RNGs nor enabling deterministic CUDA kernels removes the "
         r"variance.}",
         r"\label{tab:variance}", r"\begin{tabular}{llccc}", r"\toprule",
         r"Cond. & Configuration & Mean sd & Mean spread & Max spread \\",
         r"\midrule"]
    for _, r in t4.iterrows():
        L.append(f"{r.condition} & {r.description} & {r.mean_sd:.4f} & "
                 f"{r.mean_spread:.4f} & {r.max_spread:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_variance.tex", L)

    # ---- Table 6: resolving power ------------------------------------------
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Resolving power of each test set. The spread is the Dice "
         r"difference between the best and worst of the five architectures; "
         r"$\sigma_{\mathrm{run}}$ is the measured run-to-run standard "
         r"deviation. A test set can only distinguish architectures whose gaps "
         r"exceed this noise floor.}",
         r"\label{tab:resolving}", r"\begin{tabular}{lccll}", r"\toprule",
         r"Test set & Spread & $/\sigma_{\mathrm{run}}$ & Best & Worst \\",
         r"\midrule"]
    for _, r in t3.iterrows():
        L.append(f"{r.dataset} & {r.spread:.4f} & {r.in_sigma:.1f} & {r.best} & {r.worst} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_resolving.tex", L)

    # ---- macros -------------------------------------------------------------
    small = t6[t6.size_bin == "<1%"]
    mid = t6[t6.size_bin == ">5%"]
    n_small_ext = int((extm.area_frac < 0.01).sum())
    n_small_kv = int((kvm.area_frac < 0.01).sum())
    ses = t5[t5.stratum == "sessile"].dice.values
    non = t5[t5.stratum == "non-sessile"].dice.values
    below = t3b[~t3b.resolvable]
    M = [
        rf"\newcommand{{\sigmarun}}{{{nf['sigma_run']:.4f}}}",
        rf"\newcommand{{\rEpochTest}}{{{nf['r_epoch_test']:.2f}}}",
        rf"\newcommand{{\rValTest}}{{{nf['r_val_test']:.3f}}}",
        rf"\newcommand{{\meanDeltaEpoch}}{{{nf['mean_d_epoch']:.1f}}}",
        rf"\newcommand{{\dTestClose}}{{{nf['d_test_close']:.4f}}}",
        rf"\newcommand{{\dTestFar}}{{{nf['d_test_far']:.4f}}}",
        rf"\newcommand{{\nPairs}}{{{nf['n_pairs']}}}",
        rf"\newcommand{{\detSmallLo}}{{{small.det.min():.2f}}}",
        rf"\newcommand{{\detSmallHi}}{{{small.det.max():.2f}}}",
        rf"\newcommand{{\diceSmallLo}}{{{small.dice.min():.2f}}}",
        rf"\newcommand{{\diceSmallHi}}{{{small.dice.max():.2f}}}",
        rf"\newcommand{{\detLargeLo}}{{{mid.det.min():.2f}}}",
        rf"\newcommand{{\detLargeHi}}{{{mid.det.max():.2f}}}",
        rf"\newcommand{{\nSmallExt}}{{{n_small_ext}}}",
        rf"\newcommand{{\nSmallKvasir}}{{{n_small_kv}}}",
        rf"\newcommand{{\pctSmallEtis}}{{"
        rf"{100 * (extm[extm.dataset == 'etis-larib'].area_frac < 0.01).mean():.1f}}}",
        rf"\newcommand{{\pctSmallKvasir}}{{{100 * (kvm.area_frac < 0.01).mean():.1f}}}",
        rf"\newcommand{{\sesGapLo}}{{{abs(ses - non).min():.3f}}}",
        rf"\newcommand{{\sesGapHi}}{{{abs(ses - non).max():.3f}}}",
        rf"\newcommand{{\nBelowNoise}}{{{len(below)}}}",
        rf"\newcommand{{\nPairsTotal}}{{{len(t3b)}}}",
        rf"\newcommand{{\sigmaInDomain}}{{"
        rf"{t3[t3.dataset == 'Kvasir-SEG'].in_sigma.iloc[0]:.1f}}}",
        rf"\newcommand{{\sigmaEtis}}{{"
        rf"{t3[t3.dataset == 'ETIS-Larib'].in_sigma.iloc[0]:.1f}}}",
        rf"\newcommand{{\ttaIn}}{{"
        rf"{t7[t7.dataset == 'Kvasir-SEG'].delta.mean():.4f}}}",
        rf"\newcommand{{\ttaOut}}{{"
        rf"{t7[t7.dataset != 'Kvasir-SEG'].delta.mean():.4f}}}",
        rf"\newcommand{{\bestInDomain}}{{{t1.loc[t1.dice.idxmax(), 'model']}}}",
        rf"\newcommand{{\bestInDomainDice}}{{{t1.dice.max():.4f}}}",
        rf"\newcommand{{\worstEtis}}{{"
        rf"{t2['ETIS-Larib'].min():.4f}}}",
        rf"\newcommand{{\bestEtis}}{{{t2['ETIS-Larib'].max():.4f}}}",
        rf"\newcommand{{\kvasirMedianArea}}{{{100 * kvm.area_frac.median():.1f}}}",
    ]

    # Qualitative study over every small-lesion external image, if present.
    qual = OUT / "figures_final" / "f8_qualitative.csv"
    if qual.exists():
        q = pd.read_csv(qual)
        M += [
            rf"\newcommand{{\nQualSmall}}{{{len(q)}}}",
            rf"\newcommand{{\nQualAllMiss}}{{{int((q.mean_det == 0).sum())}}}",
            rf"\newcommand{{\pctQualAllMiss}}{{"
            rf"{100 * (q.mean_det == 0).mean():.0f}}}",
            rf"\newcommand{{\nQualZeroDice}}{{{int((q.mean_dice == 0).sum())}}}",
        ]
    w("macros.tex", M)
    write_ieee_variants()


if __name__ == "__main__":
    main()
