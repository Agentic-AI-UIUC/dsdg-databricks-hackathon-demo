from pyspark.sql import functions as F

NUMERIC_TYPES = ("int", "bigint", "smallint", "tinyint", "double", "float")


def is_numeric(spark_type):
    return spark_type in NUMERIC_TYPES or spark_type.startswith("decimal")


def split_feature_types(sdf, skip, high_cardinality):
    """Sort columns into numeric / categorical features; drop IDs, high-cardinality text and other types."""
    text = [c for c, t in sdf.dtypes if c not in skip and t in ("string", "boolean")]
    distinct = sdf.agg(*[F.approx_count_distinct(c).alias(c) for c in text]).first().asDict() if text else {}
    num_cols, cat_cols, dropped = [], [], []
    for col, spark_type in sdf.dtypes:
        if col in skip:
            continue
        if is_numeric(spark_type):
            num_cols.append(col)
        elif col in distinct and distinct[col] <= high_cardinality:
            cat_cols.append(col)
        else:
            dropped.append(col)
    return num_cols, cat_cols, dropped


def spark_prep(sdf, num_cols, cat_cols, keep=()):
    """One feature contract for training and scoring: numerics as double, categoricals as text with nulls -> 'missing'."""
    def col(c):
        return F.col(c) if c in sdf.columns else F.lit(None)

    return sdf.select(
        *keep,
        *[F.expr(f"try_cast(`{c}` AS DOUBLE)").alias(c) if c in sdf.columns else F.lit(None).cast("double").alias(c) for c in num_cols],
        *[F.coalesce(col(c).cast("string"), F.lit("missing")).alias(c) for c in cat_cols],
    )
