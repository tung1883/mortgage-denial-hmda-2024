"""Interpretation outputs: odds ratios, adjusted rates, heatmaps, thresholds, model comparison.

All numbers are associations from observational data, not causal effects.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import features as F  # noqa: E402
from fit_sklearn import load_data  # noqa: E402

R, FIG = C.RESULTS_DIR, C.FIGURES_DIR
JUMP = 0.05
ORDERED_DTI = [b for b in C.DTI_BINS if b != "Missing"]
ORDERED_CLTV = [b for b in C.CLTV_BINS if b != "Missing"]


def caption(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(bottom=0.2)


def predict(coef_csv, base, scaler, rare, model_id, include_loan_type=True, states=None):
    t = pd.read_csv(coef_csv)
    cols = t["feature"].tolist()[1:]
    assert cols == F.feature_columns(model_id, rare, states=states, include_loan_type=include_loan_type)
    X = F.build_design_pd(base, scaler, rare, cols, fine=(model_id == "D"))
    z = X @ t["coefficient"].to_numpy()[1:] + t["coefficient"].iloc[0]
    return 1 / (1 + np.exp(-z))


def with_bins(base, dti=None, cltv=None):
    b = base.copy()
    if dti is not None:
        b["dti_bin"] = dti
    if cltv is not None:
        b["cltv_bin"] = cltv
    b["dti_x_cltv_bin"] = b["dti_bin"] + "|" + b["cltv_bin"]
    return b


def odds_ratios():
    t = pd.read_csv(R / "coef_A_sklearn.csv")
    t = t[t["feature"] != "intercept"].copy()
    t.to_csv(R / "odds_ratios_A.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t["odds_ratio"], y, xerr=[t["odds_ratio"] - t["or_ci_low"], t["or_ci_high"] - t["odds_ratio"]],
                fmt="o", color="#1f4e79", capsize=2)
    ax.axvline(1, color="grey", ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(t["feature"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio of denial (log scale; 1 = same as reference level)")
    ax.set_title("Model A: odds ratios for denial with 95% CIs")
    caption(fig, "HMDA 2024 home-purchase first-lien site-built. Reference: Conventional, DTI <36, CLTV <=80. "
                 "log_income and log_loan_amount are per 1 SD. Associations, not causal effects.")
    fig.savefig(FIG / "forest_A.png", dpi=150)
    plt.close(fig)


def adjusted_rates(base, scaler, rare, kind):
    rows = []
    levels = C.DTI_BINS if kind == "dti" else C.CLTV_BINS
    for lt in C.LOAN_TYPE_LABELS:
        sub = base[base["loan_type_label"] == lt]
        for lv in levels:
            b = with_bins(sub, dti=lv) if kind == "dti" else with_bins(sub, cltv=lv)
            p = predict(R / "coef_C_sklearn.csv", b, scaler, rare, "C")
            rows.append({"loan_type": lt, f"{kind}_bin": lv, "adjusted_denial_rate": float(p.mean()),
                         "observed_n_in_bin": int((sub[f"{kind}_bin"] == lv).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(R / f"adjusted_rate_{kind}_by_program.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lt, g in out[out[f"{kind}_bin"] != "Missing"].groupby("loan_type", sort=False):
        ax.plot(g[f"{kind}_bin"], 100 * g["adjusted_denial_rate"], marker="o", label=lt)
    ax.set_xlabel("DTI bin (%)" if kind == "dti" else "CLTV bin (%)")
    ax.set_ylabel("Adjusted denial rate (%)")
    ax.set_title(f"Adjusted denial rate by {kind.upper()} bin and loan program (Model C)")
    ax.legend(title="Loan type")
    caption(fig, "Each program's rows set to every bin, other features kept, predictions averaged. "
                 "'Missing' bin omitted from plot. Associations, not causal effects.")
    fig.savefig(FIG / f"adjusted_rate_{kind}.png", dpi=150)
    plt.close(fig)
    return out


def heatmaps(base, scaler, rare):
    panels = {"All programs": (base, R / "coef_B_sklearn.csv", rare, True)}
    for lt in C.LOAN_TYPE_LABELS:
        r = {"dti_bin": rare["per_program"][lt]["dti_bin"], "dti_fine": rare["per_program"][lt]["dti_fine"]}
        panels[lt] = (base[base["loan_type_label"] == lt], R / f"coef_B_{lt}_sklearn.csv", r, False)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, (sub, cf, r, inc_lt)) in zip(axes.ravel(), panels.items()):
        grid = np.full((len(C.DTI_BINS), len(C.CLTV_BINS)), np.nan)
        for i, d in enumerate(C.DTI_BINS):
            for j, c in enumerate(C.CLTV_BINS):
                n_obs = int(((sub["dti_bin"] == d) & (sub["cltv_bin"] == c)).sum())
                p = predict(cf, with_bins(sub, dti=d, cltv=c), scaler, r, "B", include_loan_type=inc_lt)
                rate = float(p.mean())
                rows.append({"panel": name, "dti_bin": d, "cltv_bin": c, "adjusted_denial_rate": rate, "observed_n": n_obs})
                if n_obs >= C.RARE_CELL_MIN:
                    grid[i, j] = rate * 100
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(C.CLTV_BINS)), C.CLTV_BINS, rotation=30)
        ax.set_yticks(range(len(C.DTI_BINS)), C.DTI_BINS)
        ax.set_xlabel("CLTV bin (%)")
        ax.set_ylabel("DTI bin (%)")
        ax.set_title(name)
        for (i, j), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, label="Adjusted denial rate (%)")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Adjusted denial rate over DTI x CLTV (Model B); cells with <500 observed rows masked")
    caption(fig, "Overall panel uses Model B; program panels use per-program Model B fits. Blank = masked. "
                 "Associations, not causal effects.")
    fig.savefig(FIG / "heatmap_dti_cltv.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(R / "adjusted_rate_dti_x_cltv.csv", index=False)


def thresholds(dti, cltv):
    rows = []
    for kind, tbl, order in (("DTI", dti, ORDERED_DTI), ("CLTV", cltv, ORDERED_CLTV)):
        col = f"{kind.lower()}_bin"
        for lt in C.LOAN_TYPE_LABELS:
            g = tbl[tbl["loan_type"] == lt].set_index(col)["adjusted_denial_rate"].reindex(order)
            diffs = g.diff()
            hit = diffs[diffs > JUMP]
            rows.append({"variable": kind, "loan_type": lt,
                         "first_bin_with_jump": hit.index[0] if len(hit) else None,
                         "jump_pp": float(100 * hit.iloc[0]) if len(hit) else None,
                         "definition": "first bin whose adjusted rate exceeds the previous bin's by >5 percentage "
                                       "points (Missing excluded); descriptive, not a statistical test"})
    pd.DataFrame(rows).to_csv(R / "threshold_table.csv", index=False)


def model_comparison():
    m = pd.read_csv(R / "metrics.csv")
    m = m[(m["engine"] == "sklearn") & (m["subgroup"] == "overall") & m["model"].isin(["A", "B", "C", "G"])]
    out = m[["model", "pr_auc", "log_loss", "roc_auc", "brier"]].sort_values("model").reset_index(drop=True)
    gap = out.loc[out["model"] == "G", "pr_auc"].iloc[0] - out.loc[out["model"] == "C", "pr_auc"].iloc[0]
    out["flag"] = ""
    if gap > 0.02:
        out.loc[out["model"] == "G", "flag"] = f"G beats C by {gap:.3f} PR-AUC (>0.02): logistic features miss structure"
    out.to_csv(R / "model_comparison.csv", index=False)
    print(out.to_string(index=False))


def main():
    FIG.mkdir(exist_ok=True)
    base = load_data()
    scaler, rare = F.load_json(C.SCALER_JSON), F.load_json(C.RARE_CELLS_JSON)
    odds_ratios()
    dti = adjusted_rates(base, scaler, rare, "dti")
    cltv = adjusted_rates(base, scaler, rare, "cltv")
    heatmaps(base, scaler, rare)
    thresholds(dti, cltv)
    model_comparison()


if __name__ == "__main__":
    main()
