import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import features as F  # noqa: E402

NO_RARE = {"dti_bin": [], "dti_fine": []}


def test_dti_mapping_all_value_types():
    s = pd.Series(["<20%", "20%-<30%", "30%-<36%", "36", "42", "43", "49", "50%-60%", ">60%",
                   "NA", "Exempt", None, "garbage", "8888", " 45 "])
    assert F.dti_bin_pd(s).tolist() == [
        "<36", "<36", "<36", "36-42", "36-42", "43-49", "43-49", "50-60", ">60",
        "Missing", "Missing", "Missing", "Missing", "Missing", "43-49"]
    assert F.dti_fine_pd(s).tolist()[:4] == ["<20", "20-29", "30-35", "36"]
    # only garbage and 8888 are unparsable; NA/Exempt/null are expected-missing
    assert F.dti_unparsable_count(s) == 2


def test_cltv_bin_edges_right_closed():
    s = pd.Series(["80", "80.01", "90", "90.5", "95", "100", "100.01", "151", "0", "-5", "NA", "Exempt", None, "30"])
    assert F.cltv_bin_pd(s).tolist() == [
        "<=80", "80-90", "80-90", "90-95", "90-95", "95-100", ">100", ">100",
        "Missing", "Missing", "Missing", "Missing", "Missing", "<=80"]


def _frame():
    return pd.DataFrame({
        "row_id": [1, 2, 3, 4],
        "loan_type": [1, 2, 3, 4],
        "debt_to_income_ratio": ["<20%", "44", "NA", ">60%"],
        "combined_loan_to_value_ratio": ["75", "96.5", "NA", "160"],
        "income": [100, 0, None, 50],
        "loan_amount": [300000, 200000, 100000, 150000],
        "state_code": ["CA", "TX", "CA", "NY"],
    })


def test_income_missing_and_log():
    b = F.base_features_pd(_frame())
    assert b["income_missing"].tolist() == [0, 1, 1, 0]
    assert b["log_income_raw"].tolist()[1:3] == [0.0, 0.0]
    assert b["log_income_raw"].iloc[0] == pytest.approx(np.log1p(100))
    assert b["cltv_gt150"].tolist() == [0, 0, 0, 1]


def test_scaler_ignores_missing_income_and_sets_zero():
    b = F.base_features_pd(_frame())
    sc = F.fit_scaler(b)
    assert sc["log_income"]["mean"] == pytest.approx(np.mean([np.log1p(100), np.log1p(50)]))
    z = F.apply_scaler_pd(b, sc)
    assert z["log_income"].tolist()[1:3] == [0.0, 0.0]


@pytest.mark.parametrize("model", ["A", "B", "C", "D", "F"])
def test_design_columns_match_spec(model):
    b = F.base_features_pd(_frame())
    sc = F.fit_scaler(b)
    cols = F.feature_columns(model, NO_RARE)
    assert len(cols) == len(set(cols))
    X = F.build_design_pd(b, sc, NO_RARE, cols, fine=(model == "D"))
    assert X.shape == (4, len(cols))


def test_model_a_column_list_exact():
    assert F.feature_columns("A", NO_RARE) == [
        "loan_type=FHA", "loan_type=VA", "loan_type=USDA",
        "dti_bin=36-42", "dti_bin=43-49", "dti_bin=50-60", "dti_bin=>60", "dti_bin=Missing",
        "cltv_bin=80-90", "cltv_bin=90-95", "cltv_bin=95-100", "cltv_bin=>100", "cltv_bin=Missing",
        "log_income", "income_missing", "log_loan_amount"]


def test_dummy_values_and_rare_collapse():
    b = F.base_features_pd(_frame())
    sc = F.fit_scaler(b)
    rare = {"dti_bin": ["43-49|95-100"], "dti_fine": []}
    cols = F.feature_columns("B", rare)
    assert "dti_x_cltv=43-49|95-100" not in cols and cols[-1] == "dti_x_cltv=other_rare"
    X = F.build_design_pd(b, sc, rare, cols).toarray()
    j = cols.index("dti_x_cltv=other_rare")
    assert X[:, j].tolist() == [0, 1, 0, 0]  # row 2: dti 44 -> 43-49, cltv 96.5 -> 95-100
    assert X[0, cols.index("loan_type=FHA")] == 0 and X[1, cols.index("loan_type=FHA")] == 1


def test_per_program_columns_have_no_loan_type():
    cols = F.feature_columns("B", NO_RARE, include_loan_type=False)
    assert not any("loan_type" in c or c.startswith("lt_x") for c in cols)
