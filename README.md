# Mortgage denial model (HMDA 2024): scikit-learn vs Spark MLlib

Logistic regression of denial for home-purchase, first-lien, site-built applications, fit twice (Pandas/scikit-learn and Spark MLlib) with identical feature definitions. See `MODEL_PLAN.md` for the full spec.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Needs a JDK for Spark (tested with 21, Spark 4.2.0). Packages are installed in `.venv` only.

Raw data: the 2024 national loan-level CSV (4,636,615,274 bytes, 12,259,199 rows) from the FFIEC data-browser API, saved as `data/raw/hmda_2024.csv`:
`https://ffiec.cfpb.gov/v2/data-browser-api/view/nationwide/csv?years=2024`.
Override the location with the `HMDA_RAW_CSV` environment variable. The 2024 file names two fields differently from the HMDA documentation (`loan_to_value_ratio` is the CLTV field, and `open-end_line_of_credit` has a hyphen); `config.RAW_RENAMES` maps them.

## Run order

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py src/cohort.py          # funnel.csv + data/analysis.parquet
& $py src/split.py           # data/split.parquet + split_check.json
& $py src/fit_sklearn.py     # models A-G + per-program B
& $py src/fit_spark.py       # models A-F
& $py src/evaluate.py        # metrics.csv, calibration.csv
& $py src/compare_engines.py # equivalence.csv
& $py src/interpret.py       # tables + figures
& $py -m pytest tests
```

LaTeX report: `report/latex/main.tex` (build data with `python report/build_latex.py`, compile with `tools/tectonic/tectonic.exe main.tex` inside `report/latex`; Tectonic is a local binary in `tools/`). PDF: `report/Mortgage_Denial_Models_Report.pdf`.

Report with equations and charts: `report/report.html` and `report/Mortgage_Denial_Models.pdf` (text in Simplified Technical English; rebuild HTML with `python report/build_report.py`, lint with `python report/ste_lint.py report/report.html`, PDF by printing the HTML from Chrome headless).

## Status vs acceptance criteria

| Item | Status |
|---|---|
| pytest | 14 passed (features, 50k-row Pandas/Spark parity, split determinism) |
| Funnel | `results/funnel.csv`. All four steps match the proposal exactly (see below) |
| Models A, B, C both engines | Done. D, E, F also done in both engines. G sklearn only |
| Per-program Model B | Done in sklearn (4 fits). **Spark per-program fits skipped** |
| `bench_fit.py` (optional, section 13) | **Skipped** |
| Equivalence | A passes everything. B and C pass every plan criterion except max coefficient difference (see below) |
| Metrics | `metrics.csv`: ROC-AUC, PR-AUC, log loss, Brier, bootstrap 95% CIs (200 resamples, seed 42), overall and by loan type. Constant baseline included |
| Interpretation | `odds_ratios_A.csv`, `adjusted_rate_{dti,cltv}_by_program.csv`, `adjusted_rate_dti_x_cltv.csv`, `threshold_table.csv`, `model_comparison.csv`, figures in `figures/` |

## Funnel

| Step | Rows | Proposal | Diff |
|---|---|---|---|
| action_taken in 1,2,3 | 8,661,779 | 8,661,779 | 0% |
| excl. reverse / open-end / business | 6,325,493 | 6,325,493 | 0% |
| home purchase + first lien + principal residence (`occupancy_type == 1`) | 3,611,998 | 3,611,998 | 0% |
| site-built (`construction_method == 1`) with 1-4 `total_units` | 3,208,974 | 3,208,974 | 0% |

The plan's literal filters (without occupancy and unit filters) gave 3,844,698 and 3,428,191, which were 6.4% / 6.8% above the proposal. Candidate extra filters were tested against the raw file (occupancy, units, preapproval, DTI/income present, conforming limit, action codes). Only principal residence at step 3, then 1-4 units at step 4, reproduced both counts exactly. Step 4's unit restriction removes 48 multifamily site-built rows and is the same as `derived_dwelling_category == "Single Family (1-4 Units):Site-Built"`. The plan allowed an occupancy filter once it reproduced the proposal's numbers. This is an inference from matching counts; the proposal text does not state these filters.

Cohort: 3,208,974 rows, denial rate 8.83%. By loan type: Conventional 7.70%, FHA 12.53%, VA 8.17%, USDA 12.20%.

## Equivalence notes

- Model A: max coefficient difference 5e-5, ROC-AUC difference 5e-8, log loss difference 3e-9.
- Models B and C: ROC-AUC (differences 8e-7 and 7e-8), log loss, score correlation and intercept pass. **Max coefficient difference fails** (0.156 and 0.194 vs tolerance 1e-3). Cause: B and C contain the main effects plus the full DTI x CLTV cross, so the design is exactly collinear. Only the 1e-6 ridge selects a solution, and the two optimizers stop at different points along that flat direction. Spark also stopped at its 200-iteration cap (`maxIter`) on Models C, D, E and F without meeting `tol`, so incomplete convergence may contribute to the C gap (plan cause (c)); this was not tested with a higher cap. The differences are equal and opposite between a main-effect and its cell dummies (for example `cltv_bin=95-100` vs `dti_x_cltv=<36|95-100`). Fitted logits still agree (max difference 3e-3 for B and 3e-2 for C, recorded as info-only, not a plan criterion; the C figure is a worst case over 641,795 rows). Tolerances were not loosened and results were not edited. Individual B/C coefficients should not be interpreted on their own; use adjusted rates.
- Iteration counts differ (L-BFGS vs Spark's solver) and are recorded, not a pass criterion.
- Spark-native `areaUnderPR` is in `metrics.csv` (`spark_native_pr_auc`); it differs from `average_precision_score` by under 1e-4 because of different interpolation.

## Headline results (test set, sklearn)

| Model | ROC-AUC | PR-AUC | Log loss |
|---|---|---|---|
| Constant baseline | 0.500 | 0.088 | 0.2986 |
| A main effects | 0.735 | 0.426 | 0.2426 |
| B + DTI x CLTV | 0.746 | 0.445 | 0.2378 |
| C + program interactions | 0.763 | 0.478 | 0.2289 |
| G gradient boosting | 0.795 | 0.524 | 0.2170 |

G beats C by 0.046 PR-AUC (above the 0.02 flag), so the logistic features miss structure. State dummies (Model E) add about +0.012 ROC-AUC over C, which hints at geography as part of that structure.

Adjusted denial rates show sharp steps by program (Model C): Conventional jumps at DTI 50-60 (about 5.4% to 65%), FHA and VA at DTI >60 (FHA 12% to 98%, VA 7.5% to 40%), and USDA at 43-49 by the 5-point rule (8.9% to 24%) then 50-60 (94%, only 425 rows). For CLTV, only VA (>100) and USDA (>100) show a jump over 5 points; Conventional and FHA show none. `threshold_table.csv` gives the first bin with a >5 percentage-point rise; it is descriptive, not a statistical test.

## Design decisions

- Explicit named dummy columns in `config.py`/`features.py`; no `OneHotEncoder`/`StringIndexer`. Both engines use the same column list and order.
- Scaler (`results/scaler.json`) and rare-cell list (`results/rare_cells.json`, cells with <500 train rows collapsed into `other_rare`) are fit once on train and reused by Spark.
- `loan_type_x_dti` / `loan_type_x_cltv` interactions are not collapsed; a few cells (for example USDA x high DTI) are tiny and their estimates are weak.
- Model F drops rows with CLTV > 150 from both train and test. Model D uses `dti_fine` in main effects and interactions. Model E uses the most common train state as reference.
- Regularization: `lam = 1e-6`; sklearn `C = 1/(lam * n_train)`; Spark `regParam = lam`, `standardization=False`. No class weights.
- Standard errors / CIs for Models A and B come from the Fisher information `X'WX` (plus the ridge term), not statsmodels (statsmodels does not take the sparse matrix). statsmodels is in `requirements.txt` but unused.
- `row_id` comes from `row_number()` over `monotonically_increasing_id()` and is persisted. It is deterministic for a fixed file and Spark partitioning, not tied to any HMDA key (none exists).
- `analysis.parquet` is written via `toPandas()` + pyarrow, which avoids the `winutils`/`HADOOP_HOME` requirement on Windows. Spark only reads Parquet.
- The Spark fit caches the joined, feature-built frame before fitting.

## Excluded from the model

Never used as inputs: `denial_reason_1..4`, `interest_rate`, `rate_spread`, `total_loan_costs`, `total_points_and_fees`, `origination_charges`, `discount_points`, `lender_credits`, `purchaser_type`, `hoepa_status`, `aus_1..5`, any `*_result` field, and anything filled only after a decision. Protected attributes are not inputs: `applicant_race_*`, `applicant_ethnicity_*`, `applicant_sex`, `applicant_age`, `derived_race`, `derived_ethnicity`, `derived_sex`, and the co-applicant equivalents. Only the 15 columns in `config.REQUIRED_COLUMNS` are read (`occupancy_type` and `total_units` are used for the cohort filter only).

## Limitations

- Associations, not causal effects.
- Credit score, assets, employment history and the lender's underwriting rules are not in the data.
- No application date, so no temporal split (stratified random 80/20, seed 42).
- Missing DTI/CLTV is non-random (for example Conventional missing-DTI rows are denied 15% vs 34% for FHA); it is modeled as its own category, never imputed.
- DTI is reported in wide bins except integers 36-49, so thresholds cannot be resolved more finely than that (Model D).
- USDA has few high-DTI rows (425 in 50-60, 423 in >60 in the cohort); its estimates are weak.
- The cohort restricts to principal residences with 1-4 units, inferred from matching the proposal's counts.
