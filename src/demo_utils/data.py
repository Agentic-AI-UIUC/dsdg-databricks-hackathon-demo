import re

from pyspark.sql.types import StringType, StructField, StructType


def clean(name):
    """Delta-safe lower_snake_case column name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() if name else None


def read_csv(spark, path, id_col):
    """Read a Kaggle CSV with inferred types, keeping the ID as text so values like 00042 survive."""
    opts = {"header": True, "escape": '"'}
    inferred = spark.read.options(inferSchema=True, **opts).csv(path).schema
    schema = StructType([
        StructField(f.name, StringType() if clean(f.name) == id_col else f.dataType, True) for f in inferred.fields
    ])
    df = spark.read.options(**opts).schema(schema).csv(path)
    return df.toDF(*[clean(c) for c in df.columns])
