"""Pandas vs Spark parity on a 50k-row sample of analysis.parquet (skipped if the file is absent)."""
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
from split import make_split  # noqa: E402

pytestmark = pytest.mark.skipif(not C.ANALYSIS_PARQUET.exists(), reason="analysis.parquet not built")


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    df = pd.read_parquet(C.ANALYSIS_PARQUET, columns=["row_id", "denied"] + C.REQUIRED_COLUMNS)
    df = df.sample(50_000, random_state=C.SEED).reset_index(drop=True)
    sp = make_split(df.assign(loan_type=df["loan_type"]))
    df = df.merge(sp, on="row_id")
    path = tmp_path_factory.mktemp("sample") / "s.parquet"
    df.to_parquet(path, index=False)
    return df, path


@pytest.fixture(scope="module")
def spark():
    from cohort import get_spark
    s = get_spark("parity")
    yield s
    s.stop()


def test_split_deterministic():
    df = pd.read_parquet(C.ANALYSIS_PARQUET, columns=["row_id", "denied", "loan_type"]).head(200_000)
    a, b = make_split(df), make_split(df)
    assert a.equals(b)


def test_design_matrices_identical_and_model_a_matches(sample, spark):
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as SF

    df, path = sample
    base = F.base_features_pd(df)
    base["is_test"] = df["is_test"].to_numpy()
    tr = base[~base["is_test"]]
    scaler = F.fit_scaler(tr)
    rare = F.compute_rare_cells(tr)
    cols = F.feature_columns("B", rare)
    Xp = F.build_design_pd(base, scaler, rare, cols).toarray()

    sdf = spark.read.parquet(str(path))
    sb = F.base_features_spark(sdf)
    sd = F.build_design_spark(sb, scaler, rare, cols).select("row_id", *cols)
    got = sd.toPandas().set_index("row_id").loc[df["row_id"].to_numpy(), cols].to_numpy()
    np.testing.assert_allclose(got, Xp, atol=1e-9)

    # Model A fit in both engines
    cols_a = F.feature_columns("A", rare)
    ya = df["denied"].to_numpy()
    mask = ~df["is_test"].to_numpy()
    from fit_sklearn import fit_logistic
    Xa = F.build_design_pd(base, scaler, rare, cols_a)
    clf = fit_logistic(Xa[mask], ya[mask])

    sa = F.build_design_spark(sb, scaler, rare, cols_a)
    sa = VectorAssembler(inputCols=cols_a, outputCol="features").transform(sa)
    sa = sa.select("row_id", SF.col("denied").cast("double").alias("label"), "is_test", "features")
    train = sa.where(~SF.col("is_test"))
    m = LogisticRegression(regParam=C.LAM, elasticNetParam=0.0, standardization=False, fitIntercept=True,
                           maxIter=C.SPARK_MAX_ITER, tol=C.TOL).fit(train)
    assert np.abs(clf.coef_[0] - m.coefficients.toArray()).max() < 1e-3
    assert abs(clf.intercept_[0] - m.intercept) < 1e-3
