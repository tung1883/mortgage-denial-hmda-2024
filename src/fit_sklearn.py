"""Fit models A-G (+ per-program Model B) with scikit-learn; save coefficients and test scores."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import features as F  # noqa: E402

LOGIT_MODELS = ["A", "B", "C", "D", "E", "F"]


def load_data():
    cols = ["row_id", "denied"] + C.REQUIRED_COLUMNS
    df = pd.read_parquet(C.ANALYSIS_PARQUET, columns=cols)
    split = pd.read_parquet(C.SPLIT_PARQUET)
    df = df.merge(split, on="row_id", how="inner", validate="one_to_one")
    assert len(df) == len(split)
    base = F.base_features_pd(df)
    base["denied"] = df["denied"].to_numpy()
    base["is_test"] = df["is_test"].to_numpy()
    base["loan_type"] = df["loan_type"].to_numpy()
    return base


def fit_logistic(X, y):
    n = X.shape[0]
    clf = LogisticRegression(
        penalty="l2", C=1.0 / (C.LAM * n), solver="lbfgs", fit_intercept=True,
        max_iter=C.SKLEARN_MAX_ITER, tol=C.TOL)
    clf.fit(X, y)
    return clf


def coef_table(clf, columns):
    return pd.DataFrame({
        "feature": ["intercept"] + list(columns),
        "coefficient": [float(clf.intercept_[0])] + clf.coef_[0].tolist()})


def fisher_inference(X, p, coefs, intercept):
    """Wald SEs / 95% CIs from the Fisher information X'WX (+ the ridge term), intercept included."""
    import scipy.sparse as sp

    n = X.shape[0]
    Xi = sp.hstack([sp.csr_matrix(np.ones((n, 1))), X], format="csr")
    w = p * (1 - p)
    H = (Xi.T @ sp.diags(w) @ Xi).toarray()
    pen = np.eye(H.shape[0]) * C.LAM * n
    pen[0, 0] = 0.0
    cov = np.linalg.inv(H + pen)
    se = np.sqrt(np.diag(cov))
    beta = np.concatenate([[intercept], coefs])
    return pd.DataFrame({
        "se": se, "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se,
        "odds_ratio": np.exp(beta), "or_ci_low": np.exp(beta - 1.96 * se), "or_ci_high": np.exp(beta + 1.96 * se)})


def save_scores(path, base_rows, p):
    pd.DataFrame({"row_id": base_rows["row_id"].to_numpy(), "y_true": base_rows["denied"].to_numpy(),
                  "p_denied": p}).to_parquet(path, index=False)


def run_logit(tag, model_id, base, scaler, rare, states=None, include_loan_type=True, extra_train_scores=False):
    """tag names output files; returns fit info dict."""
    fine = model_id == "D"
    cols = F.feature_columns(model_id, rare, states=states, include_loan_type=include_loan_type)
    tr, te = base[~base["is_test"]], base[base["is_test"]]
    Xtr = F.build_design_pd(tr, scaler, rare, cols, fine)
    Xte = F.build_design_pd(te, scaler, rare, cols, fine)
    ytr = tr["denied"].to_numpy()
    t0 = time.perf_counter()
    clf = fit_logistic(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    ptr = clf.predict_proba(Xtr)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    tbl = coef_table(clf, cols)
    if model_id in ("A", "B") and include_loan_type:
        inf = fisher_inference(Xtr, ptr, clf.coef_[0], clf.intercept_[0])
        tbl = pd.concat([tbl, inf], axis=1)
    tbl.to_csv(C.RESULTS_DIR / f"coef_{tag}_sklearn.csv", index=False)
    save_scores(C.RESULTS_DIR / f"scores_{tag}_sklearn.parquet", te, pte)
    if extra_train_scores:
        save_scores(C.RESULTS_DIR / f"scores_{tag}_sklearn_train.parquet", tr, ptr)
    return {"tag": tag, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]), "n_cols": len(cols),
            "n_iter": int(clf.n_iter_[0]), "fit_seconds": fit_s}


def run_gbm(base, states):
    """Model G: HistGradientBoosting; bins/loan_type/state as categorical codes."""
    cat_levels = {
        "loan_type_label": C.LOAN_TYPE_LABELS, "dti_bin": C.DTI_BINS, "cltv_bin": C.CLTV_BINS,
        "state_code": states}

    def design(d):
        X = pd.DataFrame(index=d.index)
        for c, lv in cat_levels.items():
            X[c] = pd.Categorical(d[c], categories=lv).codes.astype("float64")
            X.loc[X[c] < 0, c] = np.nan
        sc = F.apply_scaler_pd(d, SCALER)
        X["log_income"], X["log_loan_amount"] = sc["log_income"], sc["log_loan_amount"]
        X["income_missing"] = d["income_missing"]
        return X

    tr, te = base[~base["is_test"]], base[base["is_test"]]
    Xtr, Xte = design(tr), design(te)
    mask = [c in cat_levels for c in Xtr.columns]
    t0 = time.perf_counter()
    gbm = HistGradientBoostingClassifier(
        categorical_features=mask, max_iter=300, learning_rate=0.1, random_state=C.SEED)
    gbm.fit(Xtr, tr["denied"].to_numpy())
    fit_s = time.perf_counter() - t0
    save_scores(C.RESULTS_DIR / "scores_G_sklearn.parquet", te, gbm.predict_proba(Xte)[:, 1])
    return {"tag": "G", "n_train": len(Xtr), "n_test": len(Xte), "n_iter": int(gbm.n_iter_), "fit_seconds": fit_s}


SCALER = None


def main():
    global SCALER
    C.RESULTS_DIR.mkdir(exist_ok=True)
    base = load_data()
    tr = base[~base["is_test"]]
    SCALER = F.fit_scaler(tr)
    F.save_json(SCALER, C.SCALER_JSON)
    rare = F.compute_rare_cells(tr)
    states = tr["state_code"].value_counts().index.tolist()
    rare["per_program"] = {
        lt: F.compute_rare_cells(tr[tr["loan_type_label"] == lt]) for lt in C.LOAN_TYPE_LABELS}
    F.save_json(rare, C.RARE_CELLS_JSON)
    F.save_json(states, C.RESULTS_DIR / "states.json")
    info = []

    for m in ["A", "B", "C", "D", "E"]:
        print("fit", m, flush=True)
        info.append(run_logit(m, m, base, SCALER, rare, states=states, extra_train_scores=(m == "A")))
    print("fit F", flush=True)
    no150 = base[base["cltv_gt150"] == 0]
    info.append(run_logit("F", "C", no150, SCALER, rare))
    for lt in C.LOAN_TYPE_LABELS:
        print("fit B program", lt, flush=True)
        sub = base[base["loan_type_label"] == lt]
        r = {"dti_bin": rare["per_program"][lt]["dti_bin"], "dti_fine": rare["per_program"][lt]["dti_fine"]}
        info.append(run_logit(f"B_{lt}", "B", sub, SCALER, r, include_loan_type=False))
    print("fit G", flush=True)
    info.append(run_gbm(base, states))
    pd.DataFrame(info).to_csv(C.RESULTS_DIR / "fit_info_sklearn.csv", index=False)
    print(pd.DataFrame(info).to_string(index=False))


if __name__ == "__main__":
    main()
