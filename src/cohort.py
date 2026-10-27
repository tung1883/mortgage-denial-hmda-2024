"""Cohort filters + target. Pandas and Spark versions share the same logic.

Run as a script to build data/analysis.parquet and results/funnel.csv from the raw CSV.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


# Filters verified against the proposal's funnel: principal residence (occupancy_type == 1) at the
# home-purchase/first-lien step, then site-built with 1-4 units (= "Single Family (1-4 Units):Site-Built").
UNITS_1_4 = ["1", "2", "3", "4"]


# ---------------------------------------------------------------- pandas
def step_masks_pd(df: pd.DataFrame) -> dict:
    """Cumulative boolean masks, one per funnel step (string-typed columns)."""
    m1 = df["action_taken"].isin(["1", "2", "3"])
    m2 = m1 & (df["reverse_mortgage"] != "1") & (df["open_end_line_of_credit"] != "1") \
        & (df["business_or_commercial_purpose"] != "1")
    m3 = m2 & (df["loan_purpose"] == "1") & (df["lien_status"] == "1") & (df["occupancy_type"] == "1")
    m4 = m3 & (df["construction_method"] == "1") & df["total_units"].isin(UNITS_1_4)
    return dict(zip(C.FUNNEL_STEPS, [pd.Series(True, index=df.index), m1, m2, m3, m4]))


def apply_cohort_pd(df: pd.DataFrame) -> pd.DataFrame:
    out = df[step_masks_pd(df)["site_built"]].copy()
    out["denied"] = (out["action_taken"] == "3").astype("int8")
    return out


# ---------------------------------------------------------------- spark
def get_spark(app="mortgage"):
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master("local[*]").appName(app)
        .config("spark.driver.memory", C.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", C.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def read_raw_spark(spark, path=C.RAW_CSV_PATH):
    """Read CSV with every column as string, canonical names, only required columns."""
    sdf = spark.read.option("header", True).option("inferSchema", False).csv(str(path))
    for old, new in C.RAW_RENAMES.items():
        if old in sdf.columns:
            sdf = sdf.withColumnRenamed(old, new)
    return sdf.select(*C.REQUIRED_COLUMNS)


def step_conditions_spark():
    from pyspark.sql import functions as F

    def ne1(c):  # null-safe "!= '1'"
        return F.coalesce(F.col(c) != "1", F.lit(True))

    c1 = F.col("action_taken").isin("1", "2", "3")
    c2 = c1 & ne1("reverse_mortgage") & ne1("open_end_line_of_credit") & ne1("business_or_commercial_purpose")
    c3 = c2 & (F.col("loan_purpose") == "1") & (F.col("lien_status") == "1") & (F.col("occupancy_type") == "1")
    c4 = c3 & (F.col("construction_method") == "1") & F.col("total_units").isin(*UNITS_1_4)
    return dict(zip(C.FUNNEL_STEPS, [F.lit(True), c1, c2, c3, c4]))


def funnel_spark(sdf) -> pd.DataFrame:
    """All step counts in one pass over the data."""
    from pyspark.sql import functions as F

    conds = step_conditions_spark()
    row = sdf.agg(*[F.sum(F.when(c, 1).otherwise(0)).alias(k) for k, c in conds.items()]).first()
    rows = []
    prev = None
    for step in C.FUNNEL_STEPS:
        n = int(row[step])
        target = C.PROPOSAL_FUNNEL.get(step)
        rows.append({
            "step": step, "rows": n,
            "dropped": None if prev is None else prev - n,
            "proposal_target": target,
            "pct_diff_vs_proposal": None if target is None else round(100 * (n - target) / target, 3),
        })
        prev = n
    return pd.DataFrame(rows)


def build_analysis_spark(sdf):
    """Cohort rows + denied + deterministic row_id."""
    from pyspark.sql import Window, functions as F

    coh = sdf.where(step_conditions_spark()["site_built"])
    coh = coh.withColumn("denied", (F.col("action_taken") == "3").cast("int"))
    # monotonically_increasing_id is deterministic for a fixed file + partitioning;
    # row_number over it gives a dense 0..n-1 id. Persisted once, never recomputed.
    w = Window.orderBy(F.monotonically_increasing_id())
    return coh.withColumn("row_id", F.row_number().over(w) - 1)


def main():
    C.DATA_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    spark = get_spark("cohort")
    sdf = read_raw_spark(spark)
    funnel = funnel_spark(sdf)
    funnel.to_csv(C.RESULTS_DIR / "funnel.csv", index=False)
    print(funnel.to_string(index=False))

    analysis = build_analysis_spark(sdf)
    # toPandas + pyarrow write avoids the Hadoop winutils requirement on Windows.
    pdf = analysis.toPandas()
    pdf["row_id"] = pdf["row_id"].astype("int64")
    pdf.to_parquet(C.ANALYSIS_PARQUET, index=False)
    print("analysis rows:", len(pdf), "denial rate:", round(pdf["denied"].mean(), 4))
    print(pdf.groupby("loan_type")["denied"].agg(["size", "mean"]).round(4).to_string())
    spark.stop()


if __name__ == "__main__":
    main()
