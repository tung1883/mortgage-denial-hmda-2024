"""Binning and explicit dummy spec. Pandas and Spark versions share one column spec.

Dummy column naming:
  loan_type=<L>, dti_bin=<B>, dti_fine=<B>, cltv_bin=<B>, state=<XX>
  dti_x_cltv=<dti>|<cltv> or dti_x_cltv=other_rare
  lt_x_dti=<L>|<dti>, lt_x_cltv=<L>|<cltv>
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

OTHER_RARE = "other_rare"

# every HMDA DTI string we recognise -> fine bin
DTI_FINE_MAP = {
    **C.DTI_FINE_STRINGS,
    **{str(i): str(i) for i in range(36, 50)},
    "50%-60%": "50-60",
    ">60%": ">60",
}
FINE_TO_COARSE = {
    "<20": "<36", "20-29": "<36", "30-35": "<36",
    **{str(i): "36-42" for i in range(36, 43)},
    **{str(i): "43-49" for i in range(43, 50)},
    "50-60": "50-60", ">60": ">60", "Missing": "Missing",
}
# strings that are expected to be missing (not counted as unparsable)
DTI_KNOWN_MISSING = {"NA", "Exempt", ""}


# ---------------------------------------------------------------- pandas bins
def dti_fine_pd(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().map(DTI_FINE_MAP).fillna("Missing").astype(object)


def dti_unparsable_count(s: pd.Series) -> int:
    t = s.astype("string").str.strip()
    known = t.isin(list(DTI_FINE_MAP) + list(DTI_KNOWN_MISSING)) | t.isna()
    return int((~known).sum())


def dti_bin_pd(s: pd.Series) -> pd.Series:
    return dti_fine_pd(s).map(FINE_TO_COARSE)


def cltv_value_pd(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    return v.where(v > 0)  # <=0 -> NaN (Missing)


def cltv_bin_pd(s: pd.Series) -> pd.Series:
    v = cltv_value_pd(s)
    e = C.CLTV_EDGES
    out = np.select(
        [v.isna(), v <= e[0], v <= e[1], v <= e[2], v <= e[3]],
        ["Missing", *C.CLTV_BINS[:4]],
        default=C.CLTV_BINS[4],
    )
    return pd.Series(out, index=s.index, dtype=object)


def base_features_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived categorical/numeric columns used by every design."""
    out = pd.DataFrame(index=df.index)
    out["row_id"] = df["row_id"] if "row_id" in df else df.index
    out["loan_type_label"] = pd.to_numeric(df["loan_type"], errors="coerce").map(C.LOAN_TYPES)
    out["dti_fine"] = dti_fine_pd(df["debt_to_income_ratio"])
    out["dti_bin"] = out["dti_fine"].map(FINE_TO_COARSE)
    out["cltv_bin"] = cltv_bin_pd(df["combined_loan_to_value_ratio"])
    cltv = cltv_value_pd(df["combined_loan_to_value_ratio"])
    out["cltv_gt150"] = (cltv > C.CLTV_AUDIT_THRESHOLD).astype("int8")
    inc = pd.to_numeric(df["income"], errors="coerce")
    out["income_missing"] = (inc.isna() | (inc <= 0)).astype("int8")
    out["log_income_raw"] = np.where(out["income_missing"] == 0, np.log1p(inc.where(inc > 0, 0.0)), 0.0)
    amt = pd.to_numeric(df["loan_amount"], errors="coerce")
    out["log_loan_amount_raw"] = np.log(amt.where(amt > 0))
    out["state_code"] = df["state_code"].astype("string")
    out["dti_x_cltv_bin"] = out["dti_bin"] + "|" + out["cltv_bin"]
    out["dti_x_cltv_fine"] = out["dti_fine"] + "|" + out["cltv_bin"]
    if "denied" in df:
        out["denied"] = df["denied"]
    return out


def fit_scaler(base: pd.DataFrame) -> dict:
    inc = base.loc[base["income_missing"] == 0, "log_income_raw"]
    amt = base["log_loan_amount_raw"]
    return {
        "log_income": {"mean": float(inc.mean()), "std": float(inc.std(ddof=0))},
        "log_loan_amount": {"mean": float(amt.mean()), "std": float(amt.std(ddof=0))},
    }


def apply_scaler_pd(base: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=base.index)
    s = scaler["log_income"]
    out["log_income"] = np.where(base["income_missing"] == 0, (base["log_income_raw"] - s["mean"]) / s["std"], 0.0)
    s = scaler["log_loan_amount"]
    out["log_loan_amount"] = (base["log_loan_amount_raw"] - s["mean"]) / s["std"]
    return out


# ---------------------------------------------------------------- rare cells
def cell_levels(fine: bool):
    d = C.DTI_FINE_BINS if fine else C.DTI_BINS
    ref_d = C.DTI_FINE_REF if fine else C.DTI_REF
    return [f"{a}|{b}" for a, b in itertools.product(d, C.CLTV_BINS)], f"{ref_d}|{C.CLTV_REF}"


def compute_rare_cells(train_base: pd.DataFrame, min_rows: int = C.RARE_CELL_MIN) -> dict:
    """Cells with < min_rows train rows (reference cell excluded), per DTI granularity."""
    res = {}
    for key, col, fine in (("dti_bin", "dti_x_cltv_bin", False), ("dti_fine", "dti_x_cltv_fine", True)):
        levels, ref = cell_levels(fine)
        counts = train_base[col].value_counts()
        res[key] = [c for c in levels if c != ref and counts.get(c, 0) < min_rows]
    return res


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------- column spec
def feature_columns(model_id: str, rare_cells: dict, states=None, include_loan_type=True) -> list:
    """Ordered dummy/numeric column names for a logistic model (A-F).

    states: ordered list of state codes with reference first (Model E).
    include_loan_type False -> per-program fits (no loan_type main effect/interactions).
    """
    fine = model_id == "D"
    dti_levels = C.DTI_FINE_BINS if fine else C.DTI_BINS
    dti_ref = C.DTI_FINE_REF if fine else C.DTI_REF
    dti_name = "dti_fine" if fine else "dti_bin"
    cols = []
    if include_loan_type:
        cols += C.dummy_cols("loan_type", C.LOAN_TYPE_LABELS, C.LOAN_TYPE_REF)
    cols += C.dummy_cols(dti_name, dti_levels, dti_ref)
    cols += C.dummy_cols("cltv_bin", C.CLTV_BINS, C.CLTV_REF)
    cols += ["log_income", "income_missing", "log_loan_amount"]
    if model_id == "A":
        return cols
    levels, ref = cell_levels(fine)
    rare = set(rare_cells["dti_fine" if fine else "dti_bin"])
    cells = [f"dti_x_cltv={c}" for c in levels if c != ref and c not in rare]
    if rare:
        cells.append(f"dti_x_cltv={OTHER_RARE}")
    cols += cells
    if model_id == "B" or not include_loan_type:
        return cols
    lts = [t for t in C.LOAN_TYPE_LABELS if t != C.LOAN_TYPE_REF]
    cols += [f"lt_x_dti={t}|{d}" for t in lts for d in dti_levels if d != dti_ref]
    cols += [f"lt_x_cltv={t}|{c}" for t in lts for c in C.CLTV_BINS if c != C.CLTV_REF]
    if model_id == "E":
        cols += [f"state={s}" for s in states[1:]]
    return cols


# ---------------------------------------------------------------- pandas design
def _column_values(name: str, base: pd.DataFrame, scaled: pd.DataFrame, rare_cells: dict, fine: bool):
    """Return a boolean mask or float array for one column spec name."""
    if name in ("log_income", "log_loan_amount"):
        return scaled[name].to_numpy()
    if name == "income_missing":
        return base["income_missing"].to_numpy()
    kind, level = name.split("=", 1)
    if kind == "loan_type":
        return (base["loan_type_label"] == level).to_numpy()
    if kind in ("dti_bin", "dti_fine", "cltv_bin"):
        return (base[kind] == level).to_numpy()
    if kind == "state":
        return (base["state_code"] == level).fillna(False).to_numpy(dtype=bool)
    if kind == "dti_x_cltv":
        cell = base["dti_x_cltv_fine" if fine else "dti_x_cltv_bin"]
        rare = set(rare_cells["dti_fine" if fine else "dti_bin"])
        if level == OTHER_RARE:
            return cell.isin(rare).to_numpy()
        return (cell == level).to_numpy()
    if kind == "lt_x_dti":
        t, d = level.split("|")
        return ((base["loan_type_label"] == t) & (base["dti_fine" if fine else "dti_bin"] == d)).to_numpy()
    if kind == "lt_x_cltv":
        t, c = level.split("|")
        return ((base["loan_type_label"] == t) & (base["cltv_bin"] == c)).to_numpy()
    raise ValueError(name)


def build_design_pd(base: pd.DataFrame, scaler: dict, rare_cells: dict, columns: list, fine: bool = False):
    """Sparse CSR float64 matrix with columns in the given order."""
    import scipy.sparse as sp

    scaled = apply_scaler_pd(base, scaler)
    n = len(base)
    cols = []
    for name in columns:
        v = _column_values(name, base, scaled, rare_cells, fine)
        if v.dtype == bool:
            idx = np.flatnonzero(v)
            cols.append(sp.csc_matrix((np.ones(len(idx)), (idx, np.zeros(len(idx), dtype=int))), shape=(n, 1)))
        else:
            cols.append(sp.csc_matrix(np.asarray(v, dtype="float64").reshape(-1, 1)))
    return sp.hstack(cols, format="csr", dtype="float64")


# ---------------------------------------------------------------- spark design
def base_features_spark(sdf):
    """Spark equivalent of base_features_pd (same output column names)."""
    from pyspark.sql import functions as F

    lt_map = F.create_map(*[x for k, v in C.LOAN_TYPES.items() for x in (F.lit(k), F.lit(v))])
    dti_map = F.create_map(*[x for k, v in DTI_FINE_MAP.items() for x in (F.lit(k), F.lit(v))])
    coarse_map = F.create_map(*[x for k, v in FINE_TO_COARSE.items() for x in (F.lit(k), F.lit(v))])
    e = C.CLTV_EDGES
    cltv_raw = F.col("combined_loan_to_value_ratio").try_cast("double")
    cltv = F.when(cltv_raw > 0, cltv_raw)
    inc = F.col("income").try_cast("double")
    amt = F.col("loan_amount").try_cast("double")
    income_missing = (inc.isNull() | (inc <= 0)).cast("int")

    out = (
        sdf.withColumn("loan_type_label", lt_map[F.col("loan_type").try_cast("int")])
        .withColumn("dti_fine", F.coalesce(dti_map[F.trim(F.col("debt_to_income_ratio"))], F.lit("Missing")))
        .withColumn("dti_bin", coarse_map[F.col("dti_fine")])
        .withColumn(
            "cltv_bin",
            F.when(cltv.isNull(), "Missing")
            .when(cltv <= e[0], C.CLTV_BINS[0])
            .when(cltv <= e[1], C.CLTV_BINS[1])
            .when(cltv <= e[2], C.CLTV_BINS[2])
            .when(cltv <= e[3], C.CLTV_BINS[3])
            .otherwise(C.CLTV_BINS[4]),
        )
        .withColumn("cltv_gt150", F.coalesce((cltv > C.CLTV_AUDIT_THRESHOLD).cast("int"), F.lit(0)))
        .withColumn("income_missing", income_missing)
        .withColumn("log_income_raw", F.when(income_missing == 0, F.log1p(inc)).otherwise(F.lit(0.0)))
        .withColumn("log_loan_amount_raw", F.when(amt > 0, F.log(amt)))
        .withColumn("dti_x_cltv_bin", F.concat_ws("|", "dti_bin", "cltv_bin"))
        .withColumn("dti_x_cltv_fine", F.concat_ws("|", "dti_fine", "cltv_bin"))
    )
    return out


def build_design_spark(sdf_base, scaler: dict, rare_cells: dict, columns: list, fine: bool = False):
    """Adds the named dummy columns (in order) to a base_features_spark frame."""
    from pyspark.sql import functions as F

    s_inc, s_amt = scaler["log_income"], scaler["log_loan_amount"]
    out = sdf_base
    out = out.withColumn(
        "log_income",
        F.when(F.col("income_missing") == 0, (F.col("log_income_raw") - s_inc["mean"]) / s_inc["std"]).otherwise(F.lit(0.0)),
    ).withColumn("log_loan_amount", (F.col("log_loan_amount_raw") - s_amt["mean"]) / s_amt["std"])
    dti_col = "dti_fine" if fine else "dti_bin"
    cell_col = "dti_x_cltv_fine" if fine else "dti_x_cltv_bin"
    rare = rare_cells["dti_fine" if fine else "dti_bin"]
    for name in columns:
        if name in ("log_income", "log_loan_amount", "income_missing"):
            continue
        kind, level = name.split("=", 1)
        if kind == "loan_type":
            cond = F.col("loan_type_label") == level
        elif kind in ("dti_bin", "dti_fine", "cltv_bin"):
            cond = F.col(kind) == level
        elif kind == "state":
            cond = F.col("state_code") == level
        elif kind == "dti_x_cltv":
            cond = F.col(cell_col).isin(rare) if level == OTHER_RARE else F.col(cell_col) == level
        elif kind == "lt_x_dti":
            t, d = level.split("|")
            cond = (F.col("loan_type_label") == t) & (F.col(dti_col) == d)
        elif kind == "lt_x_cltv":
            t, c = level.split("|")
            cond = (F.col("loan_type_label") == t) & (F.col("cltv_bin") == c)
        else:
            raise ValueError(name)
        out = out.withColumn(name, F.coalesce(cond, F.lit(False)).cast("double"))
    return out
