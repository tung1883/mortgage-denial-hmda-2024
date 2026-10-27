"""Build data/split.parquet once: stratified 80/20 on denied x loan_type, seed 42."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from features import save_json  # noqa: E402


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    strata = df["denied"].astype(str) + "_" + df["loan_type"].astype(str)
    train_idx, test_idx = train_test_split(
        df.index, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=strata)
    is_test = pd.Series(False, index=df.index)
    is_test.loc[test_idx] = True
    return pd.DataFrame({"row_id": df["row_id"].astype("int64"), "is_test": is_test})


def check_split(df: pd.DataFrame, split: pd.DataFrame) -> dict:
    j = df[["row_id", "denied"]].merge(split, on="row_id", how="inner")
    test_frac = float(j["is_test"].mean())
    tr, te = j.loc[~j["is_test"], "denied"].mean(), j.loc[j["is_test"], "denied"].mean()
    res = {
        "rows_analysis": int(len(df)), "rows_joined": int(len(j)),
        "test_fraction": test_frac,
        "denial_rate_train": float(tr), "denial_rate_test": float(te),
        "pass_rows_equal": bool(len(df) == len(j)),
        "pass_test_fraction": bool(abs(test_frac - C.TEST_SIZE) <= 0.002),
        "pass_denial_rate": bool(abs(tr - te) <= 0.001),
    }
    assert res["pass_rows_equal"] and res["pass_test_fraction"] and res["pass_denial_rate"], res
    return res


def main():
    df = pd.read_parquet(C.ANALYSIS_PARQUET, columns=["row_id", "denied", "loan_type"])
    split = make_split(df)
    split.to_parquet(C.SPLIT_PARQUET, index=False)
    save_json(check_split(df, split), C.RESULTS_DIR / "split_check.json")
    print(pd.Series(check_split(df, split)).to_string())


if __name__ == "__main__":
    main()
