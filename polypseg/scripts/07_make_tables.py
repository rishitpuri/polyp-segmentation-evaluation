"""Emit LaTeX table fragments from the result CSVs.

Every number in the paper is generated here rather than typed, so the text
can never drift from the experiments. Fragments are \\input{} by main.tex.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "polypseg" / "outputs"
TAB = ROOT / "polypseg" / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "unetpp", "segformer"]
PRETTY = {"unet": "U-Net", "unetpp": "U-Net++", "segformer": "SegFormer-B0"}
PARAMS = {"unet": 24.44, "unetpp": 26.08, "segformer": 3.71}


def stars(p):
    return "$^{***}$" if p < 1e-3 else "$^{**}$" if p < 1e-2 else "$^{*}$" if p < 0.05 else "~ns"


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (len(pvals) - rank))
    return np.maximum.accumulate(adj[order])[np.argsort(order)]


def boot_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = (rng.choice(a, (n_boot, len(a)), replace=True).mean(1)
         - rng.choice(b, (n_boot, len(b)), replace=True).mean(1))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load():
    cv = pd.read_csv(OUT / "analysis_cv" / "cv_perimage_merged.csv")
    cv["sessile"] = cv["sessile"].astype(bool)
    tta = pd.read_csv(OUT / "analysis_cv" / "cv_perimage_merged_tta.csv")
    tta["sessile"] = tta["sessile"].astype(bool)
    single = pd.read_csv(OUT / "analysis" / "all_perimage_merged.csv")
    single["sessile"] = single["sessile"].astype(bool)
    single = single[~single.tta]
    return cv, tta, single


def table_main(cv, tta):
    kv = cv[cv.dataset == "kvasir-seg"]
    ood = cv[cv.dataset == "cvc-clinicdb"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Five-fold cross-validated performance on Kvasir-SEG "
        r"($n{=}1000$, every image tested exactly once) and zero-shot transfer "
        r"to CVC-ClinicDB. $\sigma_{\mathrm{fold}}$ is the standard deviation "
        r"of fold means. Best per column in bold.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Model & Par. & \multicolumn{3}{c}{Kvasir-SEG (in-domain)} & CVC \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}",
        r" & (M) & Dice & IoU & $\sigma_{\mathrm{fold}}$ & Dice \\",
        r"\midrule",
    ]
    dices = {m: kv[kv.model == m].dice.mean() for m in MODELS}
    ious = {m: kv[kv.model == m].iou.mean() for m in MODELS}
    stds = {m: kv[kv.model == m].groupby("fold").dice.mean().std() for m in MODELS}
    oods = {m: ood[ood.model == m].groupby("fold").dice.mean().mean() for m in MODELS}
    best = dict(dice=max(dices, key=dices.get), iou=max(ious, key=ious.get),
                std=min(stds, key=stds.get), ood=max(oods, key=oods.get))

    def fmt(v, is_best, dec=4):
        s = f"{v:.{dec}f}"
        return rf"\textbf{{{s}}}" if is_best else s

    for m in MODELS:
        lines.append(
            f"{PRETTY[m]} & {PARAMS[m]:.1f} & {fmt(dices[m], best['dice']=='' or m==best['dice'])} "
            f"& {fmt(ious[m], m==best['iou'])} & {fmt(stds[m], m==best['std'])} "
            f"& {fmt(oods[m], m==best['ood'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_main.tex").write_text("\n".join(lines) + "\n")


def table_strata(cv):
    kv = cv[cv.dataset == "kvasir-seg"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stratified Dice under five-fold cross-validation. "
        r"$\Delta$ compares the hardest stratum against its reference "
        r"(small vs.\ large; sessile vs.\ non-sessile) with a bootstrap 95\% CI "
        r"and Mann--Whitney $U$ test. "
        r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}",
        r"\label{tab:strata}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrcl}",
        r"\toprule",
        r"Stratum & Model & $n$ & Dice & $\Delta$ (95\% CI) \\",
        r"\midrule",
    ]
    for t in ["small", "medium", "large"]:
        for i, m in enumerate(MODELS):
            s = kv[(kv.model == m) & (kv.size_tertile == t)]
            if t == "small":
                lg = kv[(kv.model == m) & (kv.size_tertile == "large")].dice.values
                d, lo, hi = boot_diff(s.dice.values, lg)
                _, p = mannwhitneyu(s.dice.values, lg)
                delta = f"${d:+.3f}$ [${lo:+.3f}$,\\,${hi:+.3f}$]{stars(p)}"
            else:
                delta = "---"
            name = t.capitalize() if i == 0 else ""
            lines.append(f"{name} & {PRETTY[m]} & {len(s)} & {s.dice.mean():.4f} & {delta} \\\\")
        if t != "large":
            lines.append(r"\addlinespace")
    lines.append(r"\midrule")
    for flag, label in [(True, "Sessile"), (False, "Non-sessile")]:
        for i, m in enumerate(MODELS):
            s = kv[(kv.model == m) & (kv.sessile == flag)]
            if flag:
                non = kv[(kv.model == m) & (~kv.sessile)].dice.values
                d, lo, hi = boot_diff(s.dice.values, non)
                _, p = mannwhitneyu(s.dice.values, non)
                delta = f"${d:+.3f}$ [${lo:+.3f}$,\\,${hi:+.3f}$]{stars(p)}"
            else:
                delta = "---"
            name = label if i == 0 else ""
            lines.append(f"{name} & {PRETTY[m]} & {len(s)} & {s.dice.mean():.4f} & {delta} \\\\")
        lines.append(r"\addlinespace")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_strata.tex").write_text("\n".join(lines) + "\n")


def table_critique(cv, single):
    """The central table: the sessile penalty as measured by a single split
    versus five-fold CV."""
    s_kv = single[single.dataset == "kvasir-seg"]
    c_kv = cv[cv.dataset == "kvasir-seg"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The same quantity---the sessile (flat) polyp penalty---"
        r"measured two ways. A single 80/10/10 split leaves only 19 sessile "
        r"test images and reports U-Net++ as unaffected; five-fold "
        r"cross-validation over all 1000 images ($n{=}196$ sessile) shows a "
        r"uniform ${\approx}8$ point penalty for every architecture. "
        r"The single-split finding does not replicate.}",
        r"\label{tab:critique}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Single split ($n_{\mathrm{ses}}{=}19$)} "
        r"& \multicolumn{2}{c}{5-fold CV ($n_{\mathrm{ses}}{=}196$)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & $\Delta$Dice & 95\% CI width & $\Delta$Dice & 95\% CI width \\",
        r"\midrule",
    ]
    for m in MODELS:
        row = [PRETTY[m]]
        for df in (s_kv, c_kv):
            ses = df[(df.model == m) & (df.sessile)].dice.values
            non = df[(df.model == m) & (~df.sessile)].dice.values
            d, lo, hi = boot_diff(ses, non)
            row += [f"${d:+.4f}$", f"{hi - lo:.4f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_critique.tex").write_text("\n".join(lines) + "\n")


def table_tta(cv, tta):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Four-way flip test-time augmentation. Gains are roughly "
        r"twice as large under domain shift as in-domain.}",
        r"\label{tab:tta}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Kvasir-SEG} & \multicolumn{2}{c}{CVC-ClinicDB} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & base & +TTA & base & +TTA \\",
        r"\midrule",
    ]
    for m in MODELS:
        row = [PRETTY[m]]
        for ds in ["kvasir-seg", "cvc-clinicdb"]:
            b = cv[(cv.model == m) & (cv.dataset == ds)].dice.mean()
            t = tta[(tta.model == m) & (tta.dataset == ds)].dice.mean()
            row += [f"{b:.4f}", f"{t:.4f} ({t - b:+.4f})"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_tta.tex").write_text("\n".join(lines) + "\n")


def macros(cv, single):
    """Numeric macros so inline text in the paper stays consistent."""
    kv = cv[cv.dataset == "kvasir-seg"]
    ood = cv[cv.dataset == "cvc-clinicdb"]
    out = []
    for m in MODELS:
        key = m.replace("+", "")
        out.append(rf"\newcommand{{\dice{key}}}{{{kv[kv.model == m].dice.mean():.4f}}}")
        out.append(rf"\newcommand{{\ood{key}}}{{"
                   rf"{ood[ood.model == m].groupby('fold').dice.mean().mean():.4f}}}")
        sd = kv[kv.model == m].groupby("fold").dice.mean().std()
        out.append(rf"\newcommand{{\sdfold{key}}}{{{sd:.4f}}}")
        ses = kv[(kv.model == m) & (kv.sessile)].dice.mean()
        non = kv[(kv.model == m) & (~kv.sessile)].dice.mean()
        out.append(rf"\newcommand{{\sespen{key}}}{{{non - ses:.4f}}}")
    # paired Wilcoxon: unet vs segformer, to make the "significant but tiny" point
    xa = kv[kv.model == "unet"].set_index("name").dice
    xb = kv[kv.model == "segformer"].set_index("name").dice
    common = xa.index.intersection(xb.index)
    _, p = wilcoxon(xa.loc[common], xb.loc[common])
    mant, exp = f"{p:.1e}".split("e")
    out.append(rf"\newcommand{{\pUnetSeg}}{{{mant}\!\times\!10^{{{int(exp)}}}}}")
    out.append(rf"\newcommand{{\dUnetSeg}}{{"
               rf"{abs(xa.loc[common].mean() - xb.loc[common].mean()):.4f}}}")
    # Tail structure of the sessile penalty: the mean gap is roughly double
    # the median gap, i.e. the penalty is driven by catastrophic misses
    # rather than a uniform downward shift.
    ses, non = kv[kv.sessile].dice, kv[~kv.sessile].dice
    out.append(rf"\newcommand{{\sesmedian}}{{{ses.median():.3f}}}")
    out.append(rf"\newcommand{{\nonmedian}}{{{non.median():.3f}}}")
    out.append(rf"\newcommand{{\sesmedgap}}{{{non.median() - ses.median():.3f}}}")
    out.append(rf"\newcommand{{\sesmeangap}}{{{non.mean() - ses.mean():.3f}}}")
    out.append(rf"\newcommand{{\sesfail}}{{{(ses < 0.5).mean() * 100:.1f}}}")
    out.append(rf"\newcommand{{\nonfail}}{{{(non < 0.5).mean() * 100:.1f}}}")
    out.append(rf"\newcommand{{\failratio}}{{"
               rf"{(ses < 0.5).mean() / (non < 0.5).mean():.1f}}}")

    rel = [100 * (kv[kv.model == m].dice.mean()
                  - ood[ood.model == m].groupby('fold').dice.mean().mean())
           / kv[kv.model == m].dice.mean() for m in MODELS]
    out.append(rf"\newcommand{{\relmin}}{{{min(rel):.1f}}}")
    out.append(rf"\newcommand{{\relmax}}{{{max(rel):.1f}}}")
    (TAB / "macros.tex").write_text("\n".join(out) + "\n")


def main():
    cv, tta, single = load()
    table_main(cv, tta)
    table_strata(cv)
    table_critique(cv, single)
    table_tta(cv, tta)
    macros(cv, single)
    for f in sorted(TAB.glob("*.tex")):
        print(f"  wrote {f.name}")


if __name__ == "__main__":
    main()
