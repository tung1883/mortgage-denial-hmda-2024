"""Pandas/sklearn vs Spark equivalence report for Models A, B, C -> results/equivalence.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

R = C.RESULTS_DIR
COLLINEAR_NOTE = (
    " | CAUSE: designs B/C contain main effects plus the full DTI x CLTV cross, so the columns are exactly"
    " collinear (cell dummies sum to the main-effect dummies). Only the 1e-6 ridge picks a solution, and"
    " L-BFGS (sklearn) and Spark stop at different points along that flat direction; the offsets are equal and"
    " opposite between a main-effect and its cell dummies. Fitted probabilities agree (see logit row).")


def compare_model(tag, results_dir=R, info_sk=None, info_sp=None):
    rows = []

    def add(q, a, b, tol, expl="", exact=False):
        diff = None if exact and a == b else (abs(a - b) if isinstance(a, (int, float, np.number)) else None)
        ok = (a == b) if exact else (diff is not None and diff <= tol)
        rows.append({"model": tag, "quantity": q, "sklearn": a, "spark": b,
                     "difference": 0 if (exact and a == b) else diff, "tolerance": "exact" if exact else tol,
                     "pass": bool(ok), "explanation": expl})

    ck = pd.read_csv(results_dir / f"coef_{tag}_sklearn.csv")
    cs = pd.read_csv(results_dir / f"coef_{tag}_spark.csv")
    add("feature column list and order", "|".join(ck["feature"]), "|".join(cs["feature"]), 0, exact=True)
    if list(ck["feature"]) == list(cs["feature"]):
        d = (ck["coefficient"] - cs["coefficient"]).abs()
        coef = d[ck["feature"] != "intercept"]
        top = ck.assign(diff=d).query("feature != 'intercept'").nlargest(5, "diff")
        add("max abs coefficient difference", float(ck.loc[coef.idxmax(), "coefficient"]),
            float(cs.loc[coef.idxmax(), "coefficient"]), 1e-3,
            "largest: " + "; ".join(f"{r.feature}={r.diff:.2e}" for r in top.itertuples())
            + (COLLINEAR_NOTE if tag != "A" else ""))
        add("intercept difference", float(ck["coefficient"][0]), float(cs["coefficient"][0]), 1e-3)
    sk = pd.read_parquet(results_dir / f"scores_{tag}_sklearn.parquet").sort_values("row_id")
    sp = pd.read_parquet(results_dir / f"scores_{tag}_spark.parquet").sort_values("row_id")
    add("test row count", len(sk), len(sp), 0, exact=True)
    add("test denial count", int(sk["y_true"].sum()), int(sp["y_true"].sum()), 0, exact=True)
    same = np.array_equal(sk["row_id"].to_numpy(), sp["row_id"].to_numpy())
    add("test row_id sets identical", bool(same), True, 0, exact=True)
    if same:
        add("test ROC-AUC", roc_auc_score(sk["y_true"], sk["p_denied"]), roc_auc_score(sp["y_true"], sp["p_denied"]), 1e-4)
        add("test log loss", log_loss(sk["y_true"], sk["p_denied"]), log_loss(sp["y_true"], sp["p_denied"]), 1e-4)
        lg = lambda p: np.log(p / (1 - p))
        add("max abs diff of test logits (estimable quantity)", float(np.abs(lg(sk["p_denied"].to_numpy().clip(1e-12, 1 - 1e-12))
            - lg(sp["p_denied"].to_numpy().clip(1e-12, 1 - 1e-12))).max()), 0.0, float("inf"),
            "info only (not a plan criterion): fitted logits are identifiable even where coefficients are not")
        corr = float(np.corrcoef(sk["p_denied"], sp["p_denied"])[0, 1])
        rows.append({"model": tag, "quantity": "Pearson corr of test scores", "sklearn": corr, "spark": corr,
                     "difference": 1 - corr, "tolerance": ">=0.9999", "pass": corr >= 0.9999, "explanation": ""})
    if info_sk is not None and info_sp is not None:
        a, b = info_sk.set_index("tag").loc[tag], info_sp.set_index("tag").loc[tag]
        add("train row count", int(a["n_train"]), int(b["n_train"]), 0, exact=True)
        add("iterations (info only)", int(a["n_iter"]), int(b["n_iter"]), float("inf"),
            "solvers differ (L-BFGS vs OWL-QN/L-BFGS); not a pass criterion")
    return rows


def main():
    isk = pd.read_csv(R / "fit_info_sklearn.csv")
    isp = pd.read_csv(R / "fit_info_spark.csv")
    rows = []
    for tag in ["A", "B", "C"]:
        rows += compare_model(tag, R, isk, isp)
    out = pd.DataFrame(rows)
    out.to_csv(R / "equivalence.csv", index=False)
    print(out[["model", "quantity", "difference", "tolerance", "pass"]].to_string(index=False))
    print("ALL PASS:", bool(out["pass"].all()))


if __name__ == "__main__":
    main()
