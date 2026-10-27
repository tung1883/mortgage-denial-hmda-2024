# Mortgage denial models (HMDA 2024)

Logistic regression of mortgage denial on HMDA 2024

- Report: [`report/report.pdf`](report/report.pdf)
- Original assignment: [`problem.pdf`](problem.pdf)
- Spec: [`MODEL_PLAN.md`](MODEL_PLAN.md)

## Key results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| A main effects | 0.735 | 0.426 |
| B + DTI×CLTV | 0.746 | 0.445 |
| C + program interactions | 0.763 | 0.478 |
| G gradient boosting | 0.795 | 0.524 |

- Denial jumps at a different DTI for each program: Conventional at 50–60, FHA and VA above 60, USDA at 43–49.
- CLTV matters mostly above 100%.
- scikit-learn and Spark agree on predictions (ROC-AUC difference < 1e-6). Models B and C coefficients differ because the design is collinear (main effects plus full DTI×CLTV cross), so read them through adjusted rates.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Needs a JDK for Spark. Download the [2024 national loan-level CSV](https://ffiec.cfpb.gov/v2/data-browser-api/view/nationwide/csv?years=2024) (4.6 GB) to `data/raw/hmda_2024.csv`.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py src/cohort.py          # filters, funnel.csv, data/analysis.parquet
& $py src/split.py           # stratified 80/20 split
& $py src/fit_sklearn.py     # models A-G
& $py src/fit_spark.py       # models A-F
& $py src/evaluate.py        # metrics.csv
& $py src/compare_engines.py # equivalence.csv
& $py src/interpret.py       # adjusted rates, figures
& $py -m pytest tests
```

Report: `python report/build_latex.py`, then compile `report/latex/main.tex` with any LaTeX engine.

## Notes

- Cohort filters match the proposal's counts exactly: principal residence at the purchase step, 1–4 units at the site-built step.
- Race, ethnicity, sex, age and all post-decision fields are excluded from the models.
- Limits: associations only (no credit score or lender rules), no application date (random split), USDA high-DTI cells are small.
- Not done: fit-time benchmark and Spark per-program fits. The parquet outputs and raw data are not in the repo.
