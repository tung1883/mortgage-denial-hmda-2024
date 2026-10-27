"""Single source of truth: paths, columns, bins, seeds, feature spec."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

RAW_CSV_PATH = Path(os.environ.get("HMDA_RAW_CSV", DATA_DIR / "raw" / "hmda_2024.csv"))
ANALYSIS_PARQUET = DATA_DIR / "analysis.parquet"
SPLIT_PARQUET = DATA_DIR / "split.parquet"
SCALER_JSON = RESULTS_DIR / "scaler.json"
RARE_CELLS_JSON = RESULTS_DIR / "rare_cells.json"

SEED = 42
TEST_SIZE = 0.20
LAM = 1e-6
RARE_CELL_MIN = 500
SPARK_MAX_ITER = 200
SKLEARN_MAX_ITER = 1000
TOL = 1e-7

# raw CSV header -> canonical (HMDA documentation) name. The 2024 public LAR file
# names the CLTV field `loan_to_value_ratio` and uses a hyphen in `open-end_line_of_credit`.
RAW_RENAMES = {
    "open-end_line_of_credit": "open_end_line_of_credit",
    "loan_to_value_ratio": "combined_loan_to_value_ratio",
}
SPARK_DRIVER_MEMORY = "10g"
SPARK_SHUFFLE_PARTITIONS = 64

FUNNEL_STEPS = [
    "all_rows",
    "action_taken_1_2_3",
    "excl_reverse_openend_business",
    "home_purchase_first_lien",
    "site_built",
]
PROPOSAL_FUNNEL = {
    "action_taken_1_2_3": 8_661_779,
    "excl_reverse_openend_business": 6_325_493,
    "home_purchase_first_lien": 3_611_998,
    "site_built": 3_208_974,
}

REQUIRED_COLUMNS = [
    "action_taken", "loan_type", "loan_purpose", "lien_status",
    "construction_method", "reverse_mortgage", "open_end_line_of_credit",
    "business_or_commercial_purpose", "occupancy_type", "total_units", "debt_to_income_ratio",
    "combined_loan_to_value_ratio", "income", "loan_amount", "state_code",
]

# loan_type code -> label; first is the reference level
LOAN_TYPES = {1: "Conventional", 2: "FHA", 3: "VA", 4: "USDA"}
LOAN_TYPE_LABELS = list(LOAN_TYPES.values())
LOAN_TYPE_REF = "Conventional"

DTI_BINS = ["<36", "36-42", "43-49", "50-60", ">60", "Missing"]
DTI_REF = "<36"
DTI_FINE_BINS = (
    ["<20", "20-29", "30-35"] + [str(i) for i in range(36, 50)] + ["50-60", ">60", "Missing"]
)
DTI_FINE_REF = "<20"

# right-closed upper edges; last bin is open-ended
CLTV_EDGES = [80.0, 90.0, 95.0, 100.0]
CLTV_BINS = ["<=80", "80-90", "90-95", "95-100", ">100", "Missing"]
CLTV_REF = "<=80"
CLTV_AUDIT_THRESHOLD = 150.0

# HMDA DTI strings that map to "<36"
DTI_LOW_STRINGS = {"<20%", "20%-<30%", "30%-<36%"}
DTI_FINE_STRINGS = {"<20%": "<20", "20%-<30%": "20-29", "30%-<36%": "30-35"}

CONTINUOUS = ["log_income", "log_loan_amount"]
MODEL_IDS = ["A", "B", "C", "D", "E", "F", "G"]


def dummy_cols(prefix, levels, ref):
    return [f"{prefix}={lv}" for lv in levels if lv != ref]


def main_effect_cols(dti_fine=False):
    dti = dummy_cols("dti_fine", DTI_FINE_BINS, DTI_FINE_REF) if dti_fine else dummy_cols("dti_bin", DTI_BINS, DTI_REF)
    return (
        dummy_cols("loan_type", LOAN_TYPE_LABELS, LOAN_TYPE_REF)
        + dti
        + dummy_cols("cltv_bin", CLTV_BINS, CLTV_REF)
        + ["log_income", "income_missing", "log_loan_amount"]
    )
