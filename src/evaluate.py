"""Metrics from saved score files only (engine-agnostic). Writes results/metrics.csv + calibration.csv."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

N_BOOT = 200


def grouped_counts(y, p):
    """Positive/negative counts per unique score, scores descending."""
    df = pd.DataFrame({"p": p, "y": y})
    g = df.groupby("p")["y"].agg(["sum", "count"]).sort_index(ascending=False)
    pos = g["sum"].to_numpy(dtype="float64")
    return pos, g["count"].to_numpy(dtype="float64") - pos


def auc_ap_grouped(pos, neg):
    """ROC-AUC (ties = 0.5) and average precision from grouped counts; matches sklearn."""
    P, N = pos.sum(), neg.sum()
    cum_pos = np.cumsum(pos)
    auc = np.sum(neg * (cum_pos - pos + 0.5 * pos)) / (P * N)
    cum_neg = np.cumsum(neg)
    precision = cum_pos / np.maximum(cum_pos + cum_neg, 1.0)
    ap = np.sum((pos / P) * precision)
    return auc, ap


def bootstrap_ci(y, p, rng):
    pos, neg = grouped_counts(y, p)
    counts = np.concatenate([pos, neg])
    probs = counts / counts.sum()
    k = len(pos)
    stats = []
    for _ in range(N_BOOT):
        s = rng.multinomial(int(counts.sum()), probs)
        stats.append(auc_ap_grouped(s[:k].astype("float64"), s[k:].astype("float64")))
    a = np.array(stats)
    lo, hi = np.percentile(a, [2.5, 97.5], axis=0)
    return lo[0], hi[0], lo[1], hi[1]


def metrics_row(model, engine, subgroup, y, p, rng, native_pr=None):
    pos, neg = grouped_counts(y, p)
    auc_g, ap_g = auc_ap_grouped(pos, neg)
    roc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    assert abs(roc - auc_g) < 1e-9 and abs(ap - ap_g) < 1e-9, "grouped metric mismatch"
    r_lo, r_hi, p_lo, p_hi = bootstrap_ci(y, p, rng)
    return {
        "model": model, "engine": engine, "subgroup": subgroup, "n": len(y), "denial_rate": float(np.mean(y)),
        "roc_auc": roc, "roc_auc_lo": r_lo, "roc_auc_hi": r_hi,
        "pr_auc": ap, "pr_auc_lo": p_lo, "pr_auc_hi": p_hi,
        "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
        "spark_native_pr_auc": native_pr,
    }


def calibration(model, engine, y, p):
    rank = pd.Series(p).rank(method="first")
    dec = pd.qcut(rank, 10, labels=False) + 1
    t = pd.DataFrame({"decile": dec, "p": p, "y": y}).groupby("decile").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
    t.insert(0, "engine", engine)
    t.insert(0, "model", model)
    return t


def main():
    rng = np.random.default_rng(C.SEED)
    ana = pd.read_parquet(C.ANALYSIS_PARQUET, columns=["row_id", "loan_type", "denied"])
    ana["loan_type"] = pd.to_numeric(ana["loan_type"]).map(C.LOAN_TYPES)
    split = pd.read_parquet(C.SPLIT_PARQUET)
    train_rate = float(ana.merge(split, on="row_id").query("not is_test")["denied"].mean())
    native = {}
    nf = C.RESULTS_DIR / "spark_native_metrics.csv"
    if nf.exists():
        native = pd.read_csv(nf).set_index("tag")["test_areaUnderPR"].to_dict()

    rows, cals = [], []
    for f in sorted(C.RESULTS_DIR.glob("scores_*_*.parquet")):
        m = re.match(r"scores_(.+)_(sklearn|spark)(_train)?\.parquet", f.name)
        if not m or m.group(3):
            continue
        model, engine = m.group(1), m.group(2)
        s = pd.read_parquet(f).merge(ana[["row_id", "loan_type"]], on="row_id", how="left")
        y, p = s["y_true"].to_numpy(), s["p_denied"].to_numpy()
        print("eval", model, engine, flush=True)
        nat = native.get(model) if engine == "spark" else None
        rows.append(metrics_row(model, engine, "overall", y, p, rng, nat))
        cals.append(calibration(model, engine, y, p))
        if "_" not in model:  # per-program models are already within one loan type
            for lt, g in s.groupby("loan_type"):
                rows.append(metrics_row(model, engine, lt, g["y_true"].to_numpy(), g["p_denied"].to_numpy(), rng))
    # constant baseline at the train denial rate, scored on the test set
    te = ana.merge(split, on="row_id").query("is_test")
    yb = te["denied"].to_numpy()
    pb = np.full(len(yb), train_rate)
    rows.append({"model": "const", "engine": "baseline", "subgroup": "overall", "n": len(yb),
                 "denial_rate": float(yb.mean()), "roc_auc": 0.5, "pr_auc": float(yb.mean()),
                 "log_loss": log_loss(yb, pb), "brier": brier_score_loss(yb, pb)})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "metrics.csv", index=False)
    pd.concat(cals).to_csv(C.RESULTS_DIR / "calibration.csv", index=False)
    print("wrote metrics.csv rows:", len(rows))


if __name__ == "__main__":
    main()
