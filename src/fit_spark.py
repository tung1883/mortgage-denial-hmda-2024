"""Fit logistic Models A-F with Spark MLlib using the same feature spec as fit_sklearn.

Per-program fits and Model G are sklearn-only (see README).
Caching: the joined, feature-built frame is cached before fitting (fit speed only; no timing claims here).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import features as F  # noqa: E402
from cohort import get_spark  # noqa: E402


def load_frame(spark, analysis=C.ANALYSIS_PARQUET, split=C.SPLIT_PARQUET):
    a = spark.read.parquet(str(analysis)).select("row_id", "denied", *C.REQUIRED_COLUMNS)
    s = spark.read.parquet(str(split))
    return F.base_features_spark(a.join(s, "row_id", "inner")).select(
        "row_id", "denied", "is_test", "loan_type_label", "dti_fine", "dti_bin", "cltv_bin", "cltv_gt150",
        "income_missing", "log_income_raw", "log_loan_amount_raw", "dti_x_cltv_bin", "dti_x_cltv_fine",
        "state_code")


def fit_model(sdf, tag, model_id, scaler, rare, states=None):
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F_

    fine = model_id == "D"
    cols = F.feature_columns(model_id, rare, states=states)
    d = F.build_design_spark(sdf, scaler, rare, cols, fine)
    d = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="error").transform(
        d.select("row_id", "denied", "is_test", "loan_type_label", *cols))
    d = d.select("row_id", F_.col("denied").cast("double").alias("label"), "is_test", "features").cache()
    train, test = d.where(~F_.col("is_test")), d.where(F_.col("is_test"))
    n_train = train.count()
    lr = LogisticRegression(
        regParam=C.LAM, elasticNetParam=0.0, standardization=False, fitIntercept=True,
        maxIter=C.SPARK_MAX_ITER, tol=C.TOL, featuresCol="features", labelCol="label")
    t0 = time.perf_counter()
    model = lr.fit(train)
    fit_s = time.perf_counter() - t0
    pd.DataFrame({"feature": ["intercept"] + cols,
                  "coefficient": [float(model.intercept)] + list(map(float, model.coefficients.toArray()))}
                 ).to_csv(C.RESULTS_DIR / f"coef_{tag}_spark.csv", index=False)
    scored = model.transform(test)
    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    native = {"tag": tag, "test_areaUnderPR": ev.setMetricName("areaUnderPR").evaluate(scored),
              "test_areaUnderROC": ev.setMetricName("areaUnderROC").evaluate(scored)}
    pred = scored.select(
        "row_id", F_.col("label").alias("y_true"), vector_to_array("probability")[1].alias("p_denied"))
    pdf = pred.toPandas()
    pdf["y_true"] = pdf["y_true"].astype("int8")
    pdf.to_parquet(C.RESULTS_DIR / f"scores_{tag}_spark.parquet", index=False)
    s = model.summary
    d.unpersist()
    return {"tag": tag, "n_train": n_train, "n_test": len(pdf), "n_cols": len(cols),
            "n_iter": int(s.totalIterations), "fit_seconds": fit_s}, native


def main():
    C.RESULTS_DIR.mkdir(exist_ok=True)
    scaler = F.load_json(C.SCALER_JSON)
    rare = F.load_json(C.RARE_CELLS_JSON)
    states = F.load_json(C.RESULTS_DIR / "states.json")
    spark = get_spark("fit_spark")
    base = load_frame(spark)
    info, native = [], []
    for tag, mid, flt in [("A", "A", False), ("B", "B", False), ("C", "C", False), ("D", "D", False),
                          ("E", "E", False), ("F", "C", True)]:
        print("fit", tag, flush=True)
        sdf = base.where(base["cltv_gt150"] == 0) if flt else base
        i, n = fit_model(sdf, tag, mid, scaler, rare, states)
        info.append(i)
        native.append(n)
    pd.DataFrame(info).to_csv(C.RESULTS_DIR / "fit_info_spark.csv", index=False)
    pd.DataFrame(native).to_csv(C.RESULTS_DIR / "spark_native_metrics.csv", index=False)
    print(pd.DataFrame(info).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
