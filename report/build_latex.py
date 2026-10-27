"""Builds report/latex/{numbers.tex, tables/*.tex, figures/*.pdf} from results/*.csv.

main.tex (hand-written text) includes these files, so no number in the text is typed by hand.
Compile: cd report/latex && tectonic main.tex   (or pdflatex/xelatex/latexmk main.tex)
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(__file__).parent / "latex"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

LT = ["Conventional", "FHA", "VA", "USDA"]
COL = {"Conventional": "#0072B2", "FHA": "#D55E00", "VA": "#009E73", "USDA": "#CC79A7"}
LS = {"Conventional": "-", "FHA": "--", "VA": "-.", "USDA": ":"}
MK = {"Conventional": "o", "FHA": "s", "VA": "D", "USDA": "^"}
NAVY, GREY, RULE = "#1d4f8c", "#55636f", "#d8dde2"

plt.rcParams.update({
    "pdf.fonttype": 42, "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.linewidth": 0.7, "grid.color": RULE, "grid.linewidth": 0.6,
})

# ------------------------------------------------------------------ data
funnel = pd.read_csv(R / "funnel.csv")
met = pd.read_csv(R / "metrics.csv")
ov = met[met.subgroup == "overall"]
sk = ov[ov.engine == "sklearn"].set_index("model")
base = ov[ov.model == "const"].iloc[0]
odds = pd.read_csv(R / "odds_ratios_A.csv")
dti = pd.read_csv(R / "adjusted_rate_dti_by_program.csv")
cltv = pd.read_csv(R / "adjusted_rate_cltv_by_program.csv")
heat = pd.read_csv(R / "adjusted_rate_dti_x_cltv.csv")
cal = pd.read_csv(R / "calibration.csv")
eq = pd.read_csv(R / "equivalence.csv")
thr = pd.read_csv(R / "threshold_table.csv")
fis = pd.read_csv(R / "fit_info_sklearn.csv").set_index("tag")
fip = pd.read_csv(R / "fit_info_spark.csv").set_index("tag")
bylt = met[(met.engine == "sklearn") & met.model.isin(["A", "C", "G"]) & (met.subgroup != "overall")]
prog = met[(met.engine == "sklearn") & met.model.str.startswith("B_")].set_index("model")
den = pd.read_parquet(ROOT / "data" / "analysis.parquet", columns=["denied", "loan_type"])
rate_lt = den.groupby(pd.to_numeric(den.loan_type)).denied.mean()
N = int(funnel.rows.iloc[-1])


def tex_pct(x, n=1):
    return f"{100 * x:.{n}f}\\%"


def dti_get(lt, b):
    return dti[(dti.loan_type == lt) & (dti.dti_bin == b)].iloc[0].adjusted_denial_rate


def cl_get(lt, b):
    return cltv[(cltv.loan_type == lt) & (cltv.cltv_bin == b)].iloc[0].adjusted_denial_rate


def orv(name):
    return odds[odds.feature == name].odds_ratio.iloc[0]


# ------------------------------------------------------------------ numbers.tex
num = {
    "NRows": f"{N:,}".replace(",", "\\,"), "NTrain": f"{int(fis.loc['A', 'n_train']):,}".replace(",", "\\,"),
    "NTest": f"{int(fis.loc['A', 'n_test']):,}".replace(",", "\\,"),
    "DenialRate": tex_pct(den.denied.mean(), 2),
    "RateConv": tex_pct(rate_lt[1], 2), "RateFHA": tex_pct(rate_lt[2], 2), "RateVA": tex_pct(rate_lt[3], 2), "RateUSDA": tex_pct(rate_lt[4], 2),
    "AROC": f"{sk.loc['A'].roc_auc:.3f}", "APR": f"{sk.loc['A'].pr_auc:.3f}",
    "CROC": f"{sk.loc['C'].roc_auc:.3f}", "CPR": f"{sk.loc['C'].pr_auc:.3f}",
    "GROC": f"{sk.loc['G'].roc_auc:.3f}", "GPR": f"{sk.loc['G'].pr_auc:.3f}",
    "EGain": f"{sk.loc['E'].roc_auc - sk.loc['C'].roc_auc:.3f}", "BasePR": f"{base.pr_auc:.3f}",
    "Gap": f"{sk.loc['G'].pr_auc - sk.loc['C'].pr_auc:.3f}",
    "DtiConvLow": tex_pct(dti_get("Conventional", "43-49")), "DtiConvHigh": tex_pct(dti_get("Conventional", "50-60"), 0),
    "DtiFhaMid": tex_pct(dti_get("FHA", "50-60"), 0), "DtiFhaHigh": tex_pct(dti_get("FHA", ">60"), 0),
    "DtiVaMid": tex_pct(dti_get("VA", "50-60"), 1), "DtiVaHigh": tex_pct(dti_get("VA", ">60"), 0),
    "DtiUsdaLow": tex_pct(dti_get("USDA", "36-42"), 0), "DtiUsdaMid": tex_pct(dti_get("USDA", "43-49"), 0),
    "ClConvLow": tex_pct(cl_get("Conventional", "<=80")), "ClConvHigh": tex_pct(cl_get("Conventional", ">100")),
    "ClVaLow": tex_pct(cl_get("VA", "95-100")), "ClVaHigh": tex_pct(cl_get("VA", ">100"), 0),
    "ORDtiHigh": f"{orv('dti_bin=>60'):.0f}", "ORDtiMid": f"{orv('dti_bin=50-60'):.1f}",
    "ORVA": f"{orv('loan_type=VA'):.2f}", "ORCltvHigh": f"{orv('cltv_bin=>100'):.2f}", "ORIncome": f"{orv('income_missing'):.1f}",
    "FitASk": f"{fis.loc['A', 'fit_seconds']:.0f}", "FitASp": f"{fip.loc['A', 'fit_seconds']:.0f}",
    "FitCSk": f"{fis.loc['C', 'fit_seconds']:.0f}", "FitCSp": f"{fip.loc['C', 'fit_seconds']:.0f}",
    "SparkIterC": str(int(fip.loc["C", "n_iter"])),
}
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()), encoding="utf-8")


# ------------------------------------------------------------------ figures
def savefig(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_funnel():
    labels = ["All 2024 HMDA rows", "Decided applications", "Excl. reverse, open-end, business",
              "Home purchase, first lien,\nprincipal residence", "Site-built, 1-4 units\n(analysis set)"]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, funnel.rows, color=[NAVY + "66"] * 4 + [NAVY], height=0.62)
    for yi, n in zip(y, funnel.rows):
        ax.text(n + 1.2e5, yi, f"{int(n):,}", va="center", fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.5e7)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.0f}M")
    ax.set_xlabel("rows")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    savefig(fig, "fig_funnel.pdf")


MODELS = [("A", "A  main effects"), ("B", r"B  + DTI$\times$CLTV"), ("C", "C  + program slopes"),
          ("D", "D  C, finer DTI"), ("E", "E  C + state"), ("F", r"F  C, CLTV$\leq$150"), ("G", "G  boosting")]


def fig_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, (metric, title, xlim) in zip(axes, [("roc_auc", "ROC-AUC", (0.72, 0.82)), ("pr_auc", "PR-AUC", (0.40, 0.55))]):
        for i, (m, lab) in enumerate(MODELS):
            r = sk.loc[m]
            c = "#D55E00" if m == "G" else NAVY
            ax.errorbar(r[metric], -i, xerr=[[r[metric] - r[metric + "_lo"]], [r[metric + "_hi"] - r[metric]]],
                        fmt="o", color=c, ms=4.5, capsize=2, lw=1.4)
            ax.text(r[metric], -i + 0.32, f"{r[metric]:.3f}", ha="center", fontsize=7.5)
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    axes[0].set_yticks([-i for i in range(len(MODELS))], [lab for _, lab in MODELS])
    axes[0].set_ylim(-len(MODELS) + 0.4, 0.8)
    savefig(fig, "fig_metrics.pdf")


def fig_calibration():
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo, hi = 0.015, 0.6
    ax.plot([lo, hi], [lo, hi], ls="--", color=GREY, lw=1)
    for m, c, mk, lab in [("A", "#0072B2", "o", "Model A"), ("C", "#009E73", "s", "Model C"), ("G", "#D55E00", "D", "Model G")]:
        g = cal[(cal.model == m) & (cal.engine == "sklearn")].sort_values("decile")
        ax.plot(g.mean_predicted, g.observed_rate, color=c, marker=mk, ms=4, lw=1, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.4]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(lambda v, _: f"{v * 100:g}%")
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("mean predicted denial rate (test-set decile)")
    ax.set_ylabel("observed denial rate")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "fig_calibration.pdf")


def fig_forest():
    d = odds[odds.feature != "intercept"].reset_index(drop=True)
    names = {"loan_type=FHA": "FHA (vs Conventional)", "loan_type=VA": "VA", "loan_type=USDA": "USDA",
             "dti_bin=36-42": "DTI 36-42 (vs $<$36)", "dti_bin=43-49": "DTI 43-49", "dti_bin=50-60": "DTI 50-60",
             "dti_bin=>60": "DTI $>$60", "dti_bin=Missing": "DTI missing",
             "cltv_bin=80-90": "CLTV 80-90 (vs $\\leq$80)", "cltv_bin=90-95": "CLTV 90-95",
             "cltv_bin=95-100": "CLTV 95-100", "cltv_bin=>100": "CLTV $>$100", "cltv_bin=Missing": "CLTV missing",
             "log_income": "log income (per SD)", "income_missing": "income missing", "log_loan_amount": "log loan amount (per SD)"}
    marks = {"dti": ("o", "#0072B2"), "cltv": ("s", "#D55E00"), "loan": ("D", "#009E73")}
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for i, r in d.iterrows():
        mk, mc = marks.get(r.feature.split("_")[0], ("^", "#CC79A7"))
        ax.errorbar(r.odds_ratio, -i, xerr=[[r.odds_ratio - r.or_ci_low], [r.or_ci_high - r.odds_ratio]], fmt=mk, color=mc, ms=4, lw=1.6, capsize=0)
        ax.text(r.or_ci_high * 1.12, -i, f"{r.odds_ratio:.2f}", va="center", fontsize=7.5)
    ax.axvline(1, color=GREY, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 25, 100], ["0.5", "1", "2", "5", "10", "25", "100"])
    ax.minorticks_off()
    ax.set_yticks([-i for i in range(len(d))], [names[f] for f in d.feature])
    ax.set_xlim(0.3, 220)
    ax.set_xlabel("odds ratio of denial (log scale)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    savefig(fig, "fig_forest.pdf")


def fig_lines(tbl, col, order, xlabel, name, ymax, step):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ncol = [c for c in tbl.columns if c.startswith("observed_n")][0]
    xs = np.arange(len(order))
    for lt in LT:
        g = tbl[tbl.loan_type == lt].set_index(col).reindex(order)
        ax.plot(xs, g.adjusted_denial_rate * 100, color=COL[lt], lw=1.8, label=lt)
        for x, v, n in zip(xs, g.adjusted_denial_rate * 100, g[ncol]):
            ax.plot(x, v, marker=MK[lt], ms=5.5, color=COL[lt], mfc="white" if n < 500 else COL[lt], mew=1.4)
    ax.set_xticks(xs, [o.replace("<=", r"$\leq$").replace("<", "$<$").replace(">", "$>$") for o in order])
    ax.set_ylim(0, ymax * 100)
    ax.set_yticks(np.arange(0, ymax * 100 + 0.1, step * 100))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("adjusted denial rate")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.13), handlelength=3)
    savefig(fig, name)


DB = ["<36", "36-42", "43-49", "50-60", ">60", "Missing"]
CB = ["<=80", "80-90", "90-95", "95-100", ">100", "Missing"]


def fig_heat():
    panels = ["All programs"] + LT
    titles = ["All programs (Model B)"] + LT
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6))
    cmap = plt.get_cmap("Blues")
    for ax, p, t in zip(axes.ravel(), panels, titles):
        d = heat[heat.panel == p].set_index(["dti_bin", "cltv_bin"])
        for i, dd in enumerate(DB):
            for j, cc in enumerate(CB):
                r = d.loc[(dd, cc)]
                if r.observed_n < 500:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////", ec=RULE, lw=0))
                    continue
                v = r.adjusted_denial_rate
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=cmap(0.08 + 0.85 * v), lw=0))
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.5 else "black")
        ax.set_xlim(-.5, 5.5)
        ax.set_ylim(5.5, -.5)
        ax.set_xticks(range(6), [c.replace("<=", r"$\leq$").replace(">", "$>$").replace("Missing", "Miss.") for c in CB], rotation=40, fontsize=6.5)
        ax.set_yticks(range(6), [c.replace("<", "$<$").replace(">", "$>$").replace("Missing", "Miss.") for c in DB], fontsize=6.5)
        ax.set_title(t, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel("DTI bin (%)", fontsize=7.5)
    axes[1, 0].set_ylabel("DTI bin (%)", fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("CLTV bin (%)", fontsize=7.5)
    axes[0, 1].set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    savefig(fig, "fig_heat.pdf")


fig_funnel()
fig_metrics()
fig_calibration()
fig_forest()
fig_lines(dti, "dti_bin", ["<36", "36-42", "43-49", "50-60", ">60"], "DTI bin (%)", "fig_dti.pdf", 1.0, 0.2)
fig_lines(cltv, "cltv_bin", ["<=80", "80-90", "90-95", "95-100", ">100"], "CLTV bin (%)", "fig_cltv.pdf", 0.30, 0.05)
fig_heat()


# ------------------------------------------------------------------ tables
def ci(r, m):
    return f"{r[m]:.3f}\\,[{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]"


def write(name, s):
    (TAB / name).write_text(s, encoding="utf-8")


rows = [f"Constant (train rate) & --- & 0.500 & {base.pr_auc:.3f} & {base.log_loss:.4f} & {base.brier:.4f} \\\\"]
for m, lab in MODELS:
    r = sk.loc[m]
    cols = f"{int(fis.loc[m, 'n_cols'])}" if m != "G" else "---"
    rows.append(f"\\textbf{{{m}}}\\ {lab[3:]} & {cols} & {ci(r, 'roc_auc')} & {ci(r, 'pr_auc')} & {r.log_loss:.4f} & {r.brier:.4f} \\\\")
write("metrics.tex", "\\begin{tabular}{@{}lrllrr@{}}\n\\toprule\nModel & Columns & ROC-AUC [95\\% CI] & PR-AUC [95\\% CI] & Log loss & Brier \\\\\n\\midrule\n"
      + "\n".join(rows).replace("\\textbf{G}\\ boosting", "\\textbf{G}\\ boosting") + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for lt in LT:
    r0 = bylt[(bylt.model == "A") & (bylt.subgroup == lt)].iloc[0]
    cells = []
    for m in ["A", "C", "G"]:
        r = bylt[(bylt.model == m) & (bylt.subgroup == lt)].iloc[0]
        cells += [f"{r.roc_auc:.3f}", f"{r.pr_auc:.3f}"]
    rows.append(f"{lt} & {int(r0.n):,} & {100 * r0.denial_rate:.1f}\\% & " .replace(",", "\\,") + " & ".join(cells) + " \\\\")
write("by_program.tex", "\\begin{tabular}{@{}lrrcccccc@{}}\n\\toprule\n & & & \\multicolumn{2}{c}{Model A} & \\multicolumn{2}{c}{Model C} & \\multicolumn{2}{c}{Model G} \\\\\n"
      "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nProgram & Test rows & Denied & ROC & PR & ROC & PR & ROC & PR \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for lt in LT:
    r = prog.loc["B_" + lt]
    rows.append(f"{lt} & {int(r.n):,} & {ci(r, 'roc_auc')} & {r.pr_auc:.3f} & {r.log_loss:.4f} \\\\".replace(f"{int(r.n):,}", f"{int(r.n):,}".replace(",", "\\,")))
write("prog_b.tex", "\\begin{tabular}{@{}lrlrr@{}}\n\\toprule\nModel B fit in one program & Test rows & ROC-AUC [95\\% CI] & PR-AUC & Log loss \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def status(r):
    return "\\textcolor{okgreen}{pass}" if r["pass"] else "\\textcolor{badred}{\\textbf{fail}}"


rows = []
for m in ["A", "B", "C"]:
    e = eq[eq.model == m].set_index("quantity")
    cf, ic, ra = e.loc["max abs coefficient difference"], e.loc["intercept difference"], e.loc["test ROC-AUC"]
    ll, cr = e.loc["test log loss"], e.loc["Pearson corr of test scores"]
    sci = lambda x: f"{x:.1e}"
    rows.append(f"\\textbf{{{m}}} & {sci(cf.difference)} {status(cf)} & {sci(ic.difference)} {status(ic)} & {sci(ra.difference)} {status(ra)} & "
                f"{sci(ll.difference)} {status(ll)} & {float(cr.sklearn):.6f} {status(cr)} & {int(fis.loc[m, 'n_iter'])} / {int(fip.loc[m, 'n_iter'])} \\\\")
write("equivalence.tex", "\\begin{tabular}{@{}lllllll@{}}\n\\toprule\nModel & Max coef.\\ diff. & Intercept diff. & ROC-AUC diff. & "
      "Log loss diff. & Score corr. & Iter.\\ sk / Spark \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
for v in ["DTI", "CLTV"]:
    for lt in LT:
        r = thr[(thr.variable == v) & (thr.loan_type == lt)].iloc[0]
        b = r.first_bin_with_jump
        has = pd.notna(b)
        bin_tex = str(b).replace(">", "$>$").replace("<", "$<$") if has else "none"
        rows.append(f"{v} & {lt} & {bin_tex} & {('+%.0f' % r.jump_pp) if has else '---'} \\\\")
write("thresholds.tex", "\\begin{tabular}{@{}llcr@{}}\n\\toprule\nVariable & Program & First bin with a rise above 5 pp & Rise (pp) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

pat = [("Loan type, DTI bin, CLTV bin, log income, income-missing flag, log loan amount", "XXXXXXX"),
       ("DTI$\\times$CLTV cross (rare cells pooled)", ".XXXXX."), ("Loan type $\\times$ DTI, loan type $\\times$ CLTV", "..XXXX."),
       ("Finer DTI (each integer 36--49)", "...X..."), ("State dummies", "....X.."), ("Drop rows with CLTV above 150", ".....X.")]
rows = [lab + " & " + " & ".join("\\textcolor{navy}{$\\bullet$}" if ch == "X" else "" for ch in p) + " \\\\" for lab, p in pat]
rows.append("Learner & " + " & ".join("GBM" if c == "G" else "LR" for c in "ABCDEFG") + " \\\\")
rows.append("Fit in & " + " & ".join("sk" if c == "G" else "sk+Sp" for c in "ABCDEFG") + " \\\\")
write("ladder.tex", "\\begin{tabular}{@{}p{6.4cm}ccccccc@{}}\n\\toprule\n & A & B & C & D & E & F & G \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("ok:", sorted(p.name for p in OUT.rglob("*") if p.is_file()))
