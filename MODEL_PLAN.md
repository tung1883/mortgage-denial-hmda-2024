# Mortgage Denial Model: Implementation Plan

Audience: an AI agent implementing this from scratch. Follow the decisions below; do not re-litigate them. Where a decision is marked **OPEN**, stop and report instead of guessing.

## 1. Goal and scope

Build a logistic-regression model of mortgage denial on HMDA 2024, implemented twice (scikit-learn and Spark MLlib) with matching definitions, and compare the two. The model serves a course project (Big Data Analysis, Pandas vs PySpark) whose research question is:

> In 2024, for home-purchase first-lien site-built applications, how do DTI and CLTV relate to denial, and how does that relationship differ across Conventional, FHA, VA and USDA loans?

The model is an explanatory and benchmarking tool, not a production underwriter. Priorities, in order:

1. Correct, reproducible, defensible.
2. Coefficients and adjusted rates that answer the research question.
3. Pandas/Spark parity, verified by script.
4. Predictive strength (secondary).

Out of scope: the descriptive analysis, plots for the main report, and the Pandas/Spark performance benchmark of the cleaning pipeline. Those belong to other parts of the project. This plan covers the optional predictive extension only. Section 13 covers an optional fit-time benchmark.

## 2. Decisions already made

| Topic | Decision |
|---|---|
| Primary model | Logistic regression, L2 with a tiny penalty, no class weights |
| Why unweighted | Coefficients stay interpretable and probabilities stay calibrated. Evaluate with threshold-free metrics |
| Benchmark model | `HistGradientBoostingClassifier` (sklearn only), to check whether the logistic model misses structure |
| Target | `denied = 1` if `action_taken == 3`, else `0` (codes 1 and 2) |
| Split | Stratified random 80/20, seed 42, computed once and shared by both engines |
| Metrics | ROC-AUC, PR-AUC, log loss, Brier score, decile calibration table |
| Protected attributes | Race, ethnicity, sex and age are NOT model inputs |
| Missing DTI/CLTV | Kept as an explicit "missing" category, never imputed or dropped |
| Scaling | Standardize continuous features with train-set mean/std, saved to JSON and reused by both engines |
| Equivalence tolerance | Not 1e-9. Iterative solvers differ. See section 11 |

## 3. Folder layout

Create everything under `test/mortgage-analysis/`:

```
test/mortgage-analysis/
  MODEL_PLAN.md            (this file)
  README.md                (how to install and run, written by the implementer)
  requirements.txt
  config.py                (single source of truth: paths, column names, bins, seeds, feature spec)
  src/
    cohort.py              (filters + target; Pandas and Spark versions)
    features.py            (binning and explicit dummy spec; Pandas and Spark versions)
    split.py               (builds split.parquet once)
    fit_sklearn.py
    fit_spark.py
    evaluate.py            (metrics computed from saved scores, engine-agnostic)
    interpret.py           (odds ratios, adjusted rates, threshold tables)
    compare_engines.py     (equivalence report)
    bench_fit.py           (optional, section 13)
  tests/
    test_features.py
    test_parity_small.py
  data/                    (gitignored; analysis.parquet, split.parquet)
  results/                 (CSV/JSON outputs)
  figures/
```

`config.py` holds every constant. No magic numbers elsewhere. Both engines import the same bin edges, category lists, reference levels, and seed.

## 4. Input contract

The raw file is the HMDA National Loan-Level Dataset 2024 (CSV, about 4.64 GB, 12,259,199 rows, 99 columns). Its location is a config value (`RAW_CSV_PATH`). Do not hard-code it. **OPEN**: ask the user for the path if it is not already in `config.py` or an environment variable.

If a prepared Parquet from the data-engineering stage already exists, use it. Otherwise, build `data/analysis.parquet` with Spark: read the CSV with an explicit schema (all columns as string, then cast), apply the cohort filters in section 5, keep only the columns below, add a `row_id`, and write unpartitioned Parquet (about 3.2M rows).

Required columns (HMDA names): `action_taken`, `loan_type`, `loan_purpose`, `lien_status`, `construction_method`, `reverse_mortgage`, `open_end_line_of_credit`, `business_or_commercial_purpose`, `debt_to_income_ratio`, `combined_loan_to_value_ratio`, `income`, `loan_amount`, `state_code`.

`row_id`: assign once in Spark with a deterministic `row_number()` ordered by a stable key. Persist it in the Parquet. Never recompute it later. HMDA has no unique record ID, so this is the join key for the split and for saved scores.

## 5. Cohort and target

Apply filters in order, and log the row count after each step to `results/funnel.csv`:

1. `action_taken` in {1, 2, 3}. Expected about 8,661,779.
2. Exclude reverse mortgages (`reverse_mortgage` == 1), open-end lines of credit (`open_end_line_of_credit` == 1), and business/commercial-purpose loans (`business_or_commercial_purpose` == 1). Expected about 6,325,493 in the proposal's funnel.
3. `loan_purpose == 1` (home purchase) and `lien_status == 1` (first lien). The proposal reports about 3,611,998 at this stage.
4. `construction_method == 1` (site-built). Expected about 3,208,974. This is the final analysis set.

Treat the proposal counts as targets, not truths. **OPEN**: the proposal's funnel text is ambiguous about which filter produces which count (for example, whether occupancy is filtered). If your counts differ by more than about 0.5%, do not tune filters until they match. Report the discrepancy and the counts you got. Do not add an occupancy filter unless it reproduces the proposal's numbers.

Target: `denied = (action_taken == 3)`. Report the overall denial rate and the rate by `loan_type`.

## 6. Feature specification

Build all features as explicit named dummy columns defined in `config.py`. Do NOT rely on sklearn `OneHotEncoder` or Spark `StringIndexer`/`OneHotEncoder`, because their category ordering and reference-level handling differ. Both engines must produce the identical column list in the identical order.

| Feature | Definition | Categories / transform | Reference level |
|---|---|---|---|
| `loan_type` | Codes 1-4 | `Conventional`, `FHA`, `VA`, `USDA` | Conventional |
| `dti_bin` | From `debt_to_income_ratio` string | `<36`, `36-42`, `43-49`, `50-60`, `>60`, `Missing` | `<36` |
| `cltv_bin` | From `combined_loan_to_value_ratio` parsed as float | `<=80`, `80-90`, `90-95`, `95-100`, `>100`, `Missing` | `<=80` |
| `log_income` | `log1p(income)` where income > 0, else 0 | Standardized | n/a |
| `income_missing` | 1 if income is null or <= 0 | Binary | 0 |
| `log_loan_amount` | `log(loan_amount)` | Standardized | n/a |

DTI mapping details: HMDA reports `<20%`, `20%-<30%`, `30%-<36%`, then integer strings `36` to `49`, then `50%-60%`, `>60%`, plus `NA` and `Exempt`. Map the first three to `<36`. Map integers 36-42 to `36-42` and 43-49 to `43-49`. Map `NA`, `Exempt`, null, and anything unparsable to `Missing`. Log counts of unparsable values.

CLTV details: parse to float. `NA`, `Exempt`, null, and values <= 0 map to `Missing`, and the <= 0 count is logged. Bins are right-closed: (-inf, 80], (80, 90], (90, 95], (95, 100], (100, inf). Keep values above 150: they land in `>100`. Add an `cltv_gt150` audit count to the report and run a sensitivity fit without those rows (section 7, Model F).

Scaling: compute mean and std of `log_income` (over rows with `income_missing == 0`) and of `log_loan_amount` on the TRAIN split only. Save to `results/scaler.json`. Apply the same values in both engines. `log_income` for missing-income rows is set to 0 after scaling, with `income_missing` carrying the signal.

Interaction features:

- `dti_x_cltv`: the 6 x 6 crossed categories, as dummies. Collapse any cell with fewer than 500 TRAIN rows into `other_rare`. Save the collapse list to `results/rare_cells.json` and apply it to test data and to Spark from that file.
- `loan_type_x_dti` and `loan_type_x_cltv`: for the program-difference question.

Never put these in the model: `denial_reason_1..4`, `interest_rate`, `rate_spread`, `total_loan_costs`, `total_points_and_fees`, `origination_charges`, `discount_points`, `lender_credits`, `purchaser_type`, `hoepa_status`, `aus_1..5`, any `*_result` fields, or anything else filled only after a decision or only for originated loans. Also exclude `applicant_race_*`, `applicant_ethnicity_*`, `applicant_sex`, `applicant_age`, `derived_race`, `derived_ethnicity`, `derived_sex`, and the co-applicant equivalents. Document the exclusion list in `README.md`.

## 7. Model set

Fit these in order. Every logistic model has a Pandas (sklearn) and a Spark version except where noted.

| ID | Features | Purpose |
|---|---|---|
| A | `loan_type`, `dti_bin`, `cltv_bin`, `log_income`, `income_missing`, `log_loan_amount` | Baseline main effects |
| B | A + `dti_x_cltv` | Does the joint DTI/CLTV pattern add anything? |
| C | B + `loan_type_x_dti` + `loan_type_x_cltv` | Program-specific slopes (research question 3) |
| D | C with `dti_bin` replaced by a finer `dti_fine` (see below) | Sensitivity for the "jump near DTI 50%" hypothesis |
| E | C + `state_code` dummies (reference: most common state) | Sensitivity for geography. Optional |
| F | C fit with `cltv_gt150` rows removed | Sensitivity for implausible CLTV |
| G | `HistGradientBoostingClassifier`, sklearn only | Benchmark. Inputs: bins as categorical, log income/amount, `loan_type`, `state_code` |

`dti_fine`: `<20`, `20-29`, `30-35`, each integer 36 to 49, `50-60`, `>60`, `Missing`. HMDA reports integers only for 36 to 49 and wide ranges elsewhere. The data cannot resolve thresholds more finely than that, and the report must say so.

Per-program fits: also fit Model B separately within each `loan_type` (4 fits, no `loan_type` features). Skip the Spark version if time is short, but state that in the README. USDA has few high-DTI rows (under 500 per group in some cells), so apply the rare-cell collapse and report that estimates there are weak.

Regularization (to make the two engines optimize the same objective):

- Spark: `LogisticRegression(regParam=lam, elasticNetParam=0.0, standardization=False, fitIntercept=True, maxIter=200, tol=1e-7)`.
- sklearn: `LogisticRegression(penalty="l2", C=1/(lam*n_train), solver="lbfgs", max_iter=1000, tol=1e-7)`.
- `lam = 1e-6` in `config.py`.

This is exact because sklearn minimizes `0.5*||w||^2 + C*sum(loss)` and Spark minimizes `mean(loss) + lam*0.5*||w||^2`. Neither penalizes the intercept. Do NOT use `class_weight`.

## 8. Train/test split

`src/split.py` runs once, in Pandas, and writes `data/split.parquet` with columns `row_id`, `is_test`.

- Stratify on `denied` x `loan_type` (8 strata).
- `test_size=0.20`, `random_state=42`.
- Both engines join `analysis.parquet` to `split.parquet` on `row_id`. Never re-split inside an engine.
- Assert after the join: row counts equal, test fraction within 0.2% of 20%, and denial rate in train vs test within 0.1 percentage points. Write these checks to `results/split_check.json`.

No temporal split is possible. HMDA 2024 here has no application date. State that in the README.

## 9. Pandas / sklearn pipeline (`fit_sklearn.py`)

1. Read `analysis.parquet` with only the needed columns (about 15). Join the split.
2. Build the design matrix via `features.py`. Use `scipy.sparse` CSR, float64. About 2.6M train rows by up to a few hundred columns, so avoid dense matrices.
3. Fit the scaler on train and save it. Fit the models in section 7.
4. For each model, save to `results/`:
   - `coef_<ID>_sklearn.csv`: feature name, coefficient, and the intercept.
   - `scores_<ID>_sklearn.parquet`: `row_id`, `y_true`, `p_denied` for the test set. Also save train scores for Model A only, for a train-vs-test gap check.
5. For Model A (and B if memory allows), fit a `statsmodels` `Logit`/`GLM(Binomial)` on the same design to get standard errors and 95% CIs, then report odds ratios. If statsmodels cannot handle the matrix size, compute standard errors from the Fisher information `X' W X` and say so. Skip inference for G.

## 10. Spark pipeline (`fit_spark.py`)

1. Session: `SparkSession.builder.master("local[*]")`. Set `spark.driver.memory` to roughly 10-12g (the machine has 31 GB), `spark.sql.shuffle.partitions` to about 64, and `spark.sql.session.timeZone` to UTC.
2. Read `analysis.parquet` and `split.parquet`, join on `row_id`, and build the same dummy columns with `when(...)`, using `features.py`'s Spark functions and the same `config.py` lists. Read `scaler.json` and `rare_cells.json`. Do not recompute them.
3. `VectorAssembler` over the fixed ordered column list, then `LogisticRegression` per section 7. Do not cache anything before timing experiments. Caching is allowed for the model fit itself, but record that it was used.
4. Save coefficients and test-set scores in the same schema as the sklearn outputs, with `_spark` in the file name. Extract `p_denied` from the probability vector's second element.
5. Windows note: Spark on Windows needs a JDK (11 or 17) and `HADOOP_HOME` with `winutils.exe` when writing local files. Check this before debugging anything else if Parquet writes fail.

## 11. Evaluation (`evaluate.py`)

Compute all metrics from saved score files, not from engine-internal evaluators, so the two engines are measured identically.

- ROC-AUC: `sklearn.metrics.roc_auc_score`.
- PR-AUC: `average_precision_score`. Also report Spark's native `areaUnderPR` as a separate column. The two use different interpolation, so small differences are expected and should be explained rather than hidden.
- Log loss, Brier score.
- Calibration: deciles of predicted probability, with mean predicted, observed denial rate, and count.
- Metrics overall and by `loan_type`.
- A baseline row: a constant predictor at the train denial rate, for context.
- Bootstrap 95% CI for ROC-AUC and PR-AUC on the test scores (200 resamples, seed 42).

Write `results/metrics.csv` with rows by model ID, engine, and subgroup.

## 12. Interpretation outputs (`interpret.py`)

These answer the research questions directly:

1. Odds ratios with 95% CIs for Model A, as a table and a forest plot.
2. Adjusted denial rate by DTI bin, per program (RQ2 and RQ3). Use Model C. For each `loan_type` and each DTI bin, set every row in that program to that bin, keep its other features, predict, and average. Output `results/adjusted_rate_dti_by_program.csv`.
3. The same for CLTV bins.
4. Heatmap of adjusted rate over DTI x CLTV (Model B), overall and per program, with cells below 500 rows masked.
5. Threshold table: for each program, the first DTI bin where the adjusted rate rises by more than 5 percentage points over the previous bin, and the same for CLTV. State this definition in the output. It is descriptive, not a statistical test.
6. Model comparison: A vs B vs C vs G on PR-AUC and log loss. If G beats C by a wide margin (more than about 0.02 PR-AUC), flag it as evidence of structure the logistic features miss.

Every figure needs a title, labeled axes, units, and a one-line caption. The README or report text must say these are associations, not causal effects, and that credit score, assets, and employment history are not in the data.

## 13. Optional: fit-time benchmark (`bench_fit.py`)

Only if the main work is finished. Measure Model A training time for sklearn vs Spark on 25%, 50% and 100% of the train set (nested samples, seed 42), with one warm-up run then the median of 5 timed runs. Time the fit only, excluding feature building, and record Spark session startup separately. Do not assume Spark is faster. Report the numbers and speedup as measured, plus the machine specs (Intel Core Ultra 5 235U, 31 GB RAM, Windows 11).

## 14. Equivalence checks (`compare_engines.py`)

Compare Pandas and Spark for Models A, B, C. Write `results/equivalence.csv` with columns: quantity, sklearn value, Spark value, difference, tolerance, pass/fail, explanation.

| Quantity | Tolerance |
|---|---|
| Train and test row counts | Exact |
| Feature column list and order | Exact |
| Denial count in train and test | Exact |
| Max absolute coefficient difference | 1e-3 (report the 5 largest differences) |
| Intercept difference | 1e-3 |
| Test ROC-AUC difference | 1e-4 |
| Test log loss difference | 1e-4 |
| Pearson correlation of test scores | At least 0.9999 |

If a check fails, investigate in this order and write down what you find: (a) a feature-definition mismatch, (b) a different column order, (c) a solver not converged (check iterations and tolerance), (d) a regularization mismatch (recheck `C = 1/(lam*n)`), (e) a data difference such as null handling or float casting. Do not loosen tolerances to pass, and do not edit results.

## 15. Tests

`tests/test_features.py` (tiny hand-built frame, no Spark): DTI string mapping for every HMDA value type including `NA`, `Exempt`, null and garbage; CLTV bin edges at 80, 90, 95, 100 (right-closed); income <= 0 handling; `income_missing` flag; the dummy column list matches `config.py` exactly.

`tests/test_parity_small.py`: build a 50,000-row random sample of `analysis.parquet` (fixed seed), run both engines' feature builders, and assert identical design matrices. Then fit Model A in both and assert the section 14 tolerances. It must run in under a few minutes.

Also add a determinism test: running `split.py` twice yields an identical `split.parquet`.

## 16. Environment and constraints

- Python 3.10+. Pin versions in `requirements.txt` (`pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `pyspark`, `matplotlib`, `pytest`).
- Windows, PowerShell/Git Bash. Use forward slashes or `pathlib`. Large intermediate files go in `data/`, never in the repo root.
- Memory: 31 GB RAM. Select only needed columns when reading. Do not load all 99 columns.
- Print aggregates and schema summaries only. Do not dump individual records to the console or logs.
- Do not run `git commit` or `git push`. The user must ask for a commit explicitly. Do not publish anything externally.
- Seeds: 42 everywhere (split, bootstrap, any subsampling). Set it in `config.py`.

## 17. Acceptance criteria

Done means all of the following hold:

1. `python -m pytest test/mortgage-analysis/tests` passes.
2. `results/funnel.csv` exists and its counts are compared with the proposal's, with any discrepancy explained.
3. Models A, B, C exist for both engines. D, F, G are present, and E and the per-program fits are either present or explicitly marked skipped in the README.
4. `results/equivalence.csv` shows every check passing, or each failure with a documented cause.
5. `results/metrics.csv` has ROC-AUC, PR-AUC, log loss and Brier score with bootstrap CIs, overall and by `loan_type`.
6. The section 12 tables and figures exist.
7. `README.md` explains setup, run order, the exclusion list, assumptions, and limitations (correlation not causation, no credit score, no application date, non-random missing DTI/CLTV, protected attributes excluded, USDA small cells).

Suggested run order: `split.py` (after `analysis.parquet` exists), `fit_sklearn.py`, `fit_spark.py`, `evaluate.py`, `compare_engines.py`, `interpret.py`, then `bench_fit.py` if wanted.

## 18. Open items to confirm with the user before or during implementation

1. Path to the raw HMDA CSV, and whether a prepared `analysis.parquet` already exists from the data-engineering stage.
2. Whether the funnel counts in section 5 reproduce. If not, which filter definition the user wants.
3. Whether to include the optional Models E and G and the section 13 benchmark, given the time left.
