# Databricks notebook source
# MAGIC %md
# MAGIC # Leverage Databricks for your hackathon: an end-to-end demo
# MAGIC
# MAGIC A simple **sample dataset** (fake shipments + regional news headlines) walked through every Databricks piece you can use
# MAGIC on your own hackathon data:
# MAGIC
# MAGIC **Volume ➜ Delta (Bronze/Silver/Gold) ➜ AI Functions on the news ➜ Spark SQL ➜ MLflow ➜ Unity Catalog ➜ Spark batch scoring ➜ Kaggle-style submission ➜ Genie & Dashboards**
# MAGIC
# MAGIC Runs on **Databricks Free Edition** (serverless). Just **Run all** (about 3–5 minutes). The sample data is generated for you on the first run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup
# MAGIC Serverless ships a slim MLflow, so we install this repo as a package (`pyproject.toml`: full MLflow for the Unity Catalog registry, Plotly, and our `demo_utils` helpers), then restart Python.

# COMMAND ----------

# MAGIC %pip install -q ..

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Widgets let a Databricks Job (or the bundle) pass these in; defaults match Free Edition.
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("volume", "demo_sample")
CATALOG, SCHEMA, VOLUME = (dbutils.widgets.get(k) for k in ("catalog", "schema", "volume"))

ID, TARGET, DATE = "shipment_id", "is_delayed", "order_date"
HIGH_CARDINALITY = 100     # drop text features with more distinct values than this
SEED = 42
AI_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # used by ai_query() for the risk briefings
NEWS_LABELS = ["port strike", "severe weather", "trade policy", "no disruption"]

MODEL_NAME = f"{CATALOG}.{SCHEMA}.shipment_delay_model"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
SUBMISSION_PATH = f"{VOLUME_PATH}/submission.csv"

# COMMAND ----------

import os
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
from pyspark.sql import functions as F

from demo_utils.data import read_csv
from demo_utils.features import is_numeric, spark_prep, split_feature_types
from demo_utils.models import candidate_models, preprocessor, score
from demo_utils.sample_data import write_sample_data
from demo_utils.viz import ORANGE, RISK_COLORS, banner_html, kpi_html, links_html

try:
    from databricks.sdk import WorkspaceClient
    HOST = WorkspaceClient().config.host.rstrip("/")
except Exception:
    HOST = ""


def T(name):
    return f"{CATALOG}.{SCHEMA}.{name}"


spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
written = write_sample_data(VOLUME_PATH)
print(f"Sample data in {VOLUME_PATH}: {sorted(os.listdir(VOLUME_PATH))}" + (f" (just wrote {written})" if written else ""))

displayHTML(banner_html(
    "Shipment Delay Risk",
    "Sample data ➜ governed lakehouse ➜ LLM news signals ➜ tracked models ➜ submission ➜ chat with your data. One notebook.",
    ["📁 Volume", "🥉 Bronze", "📰 AI on news", "🥈 Silver", "🔎 EDA", "🧪 MLflow", "🏛️ Unity Catalog",
     "⚡ Spark scoring", "🥇 Gold", "🏁 submission.csv", "💬 Genie + 📊 Dashboard"],
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bronze: raw CSV ➜ Delta tables
# MAGIC Bronze = the data exactly as it arrived, stored as **Delta** (versioned, queryable with SQL, governed by Unity Catalog).
# MAGIC `read_csv` (in `src/demo_utils/data.py`) cleans column names to `lower_snake_case` and keeps IDs as text so values like `00042` survive.
# MAGIC
# MAGIC With your own data you'd do the same for every file: one loop, one Delta table each.

# COMMAND ----------

bronze = {"bronze_train": "train.csv", "bronze_test": "test.csv", "bronze_news": "news.csv"}
for table, file_name in bronze.items():
    df = read_csv(spark, f"{VOLUME_PATH}/{file_name}", ID)
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(table))
    print(f"{table:<14} {spark.table(T(table)).count():>7,} rows x {len(df.columns)} columns")

display(spark.table(T("bronze_train")).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3a. AI Functions on the news 📰
# MAGIC The signal is in the words. `ai_classify()` runs an LLM **inside SQL** over every headline: no API keys, no extra libraries.
# MAGIC Each region-month gets a label like `port strike` or `no disruption`, saved as `silver_news_signals`.
# MAGIC Watch for decoys: some headlines are rumors that never happen, so the model has to learn how much to trust this signal.

# COMMAND ----------

labels_sql = ", ".join(f"'{label}'" for label in NEWS_LABELS)
try:
    signals = spark.sql(f"""
        SELECT region, date_format(month, 'yyyy-MM') AS month, headline, ai_classify(headline, ARRAY({labels_sql})) AS news_signal
        FROM {T('bronze_news')}
    """)
    signals.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("silver_news_signals"))
except Exception as e:
    print(f"AI Functions unavailable ({str(e).splitlines()[0][:160]}); using 'unknown' so the demo keeps going.")
    (spark.table(T("bronze_news")).select("region", F.date_format("month", "yyyy-MM").alias("month"), "headline", F.lit("unknown").alias("news_signal"))
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("silver_news_signals")))

display(spark.sql(f"SELECT news_signal, COUNT(*) AS headlines FROM {T('silver_news_signals')} GROUP BY 1 ORDER BY 2 DESC"))
display(spark.table(T("silver_news_signals")).where("news_signal != 'no disruption'").limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b. Silver: clean, engineer, join
# MAGIC The **same** function runs on train and test so their features always match:
# MAGIC - trim text, turn empty strings into nulls
# MAGIC - date parts, two derived features, and `n_missing` (how incomplete a record is)
# MAGIC - **join** that month's news signal for the shipment's region
# MAGIC - train only: drop rows with no label and exact duplicates

# COMMAND ----------


def to_silver(df, is_train):
    df = df.select([
        F.when(F.trim(c) == "", None).otherwise(F.trim(c)).alias(c) if t == "string" and c != ID else F.col(c)
        for c, t in df.dtypes
    ])
    df = (df.withColumn(DATE, F.to_date(DATE))
            .withColumn("order_month", F.month(DATE))
            .withColumn("order_dayofweek", F.dayofweek(DATE))
            .withColumn("value_per_unit", F.expr("order_value_usd / nullif(order_quantity, 0)"))
            .withColumn("is_long_haul", (F.col("distance_km") > 5000).cast("int")))
    feature_like = [c for c in df.columns if c not in (ID, TARGET)]
    df = df.withColumn("n_missing", sum((F.col(c).isNull().cast("int") for c in feature_like), F.lit(0)))

    news = spark.table(T("silver_news_signals")).select(
        F.col("region").alias("origin_region"), "month", "news_signal")
    df = (df.withColumn("month", F.date_format(DATE, "yyyy-MM"))
            .join(news, on=["origin_region", "month"], how="left")
            .drop("month")
            .fillna({"news_signal": "no news"}))
    if is_train:
        df = df.dropna(subset=[TARGET]).dropDuplicates()
    return df


for src, dst, is_train in [("bronze_train", "silver_train", True), ("bronze_test", "silver_test", False)]:
    to_silver(spark.table(T(src)), is_train).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T(dst))
    print(f"{dst}: {spark.table(T(dst)).count():,} rows")

display(spark.table(T("silver_train")).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quick EDA in Spark SQL
# MAGIC How common are delays, **where** do they cluster, and does the **news** actually predict them?

# COMMAND ----------

silver = spark.table(T("silver_train"))
stats = silver.agg(F.avg(TARGET).alias("rate"), F.avg("n_missing").alias("miss")).first()
displayHTML(kpi_html({
    "Train shipments": f"{silver.count():,}",
    "Test shipments": f"{spark.table(T('silver_test')).count():,}",
    "Headlines classified": f"{spark.table(T('silver_news_signals')).count():,}",
    "Delay rate": f"{stats['rate']:.1%}",
    "Avg missing / row": f"{stats['miss']:.2f}",
}))

# COMMAND ----------

for group_col in ("origin_region", "news_signal"):
    q = f"""
    SELECT {group_col}, COUNT(*) AS shipments, ROUND(AVG({TARGET}) * 100, 1) AS delay_rate_pct
    FROM {T('silver_train')}
    GROUP BY {group_col}
    ORDER BY delay_rate_pct DESC
    """
    px.bar(spark.sql(q).toPandas(), x=group_col, y="delay_rate_pct", text="delay_rate_pct", color="delay_rate_pct",
           color_continuous_scale="OrRd", hover_data=["shipments"], title=f"Delay rate (%) by {group_col}").show()

# COMMAND ----------

numeric = [c for c, t in silver.dtypes if is_numeric(t) and c not in (ID, TARGET)]
q_corr = "\nUNION ALL\n".join(
    f"SELECT '{c}' AS feature, ROUND(corr(CAST({c} AS DOUBLE), {TARGET}), 3) AS correlation FROM {T('silver_train')}"
    for c in numeric
)
corr = spark.sql(f"SELECT * FROM ({q_corr}) WHERE correlation IS NOT NULL ORDER BY abs(correlation)").toPandas()
px.bar(corr, x="correlation", y="feature", orientation="h", color="correlation", color_continuous_scale="RdBu_r",
       range_color=[-1, 1], title="Which numeric signals move with delays?").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train 3 models, tracked with MLflow
# MAGIC - **Spark** picks and preps the features (`src/demo_utils/features.py`): drop the ID and very-high-cardinality text, cast numerics, fill missing text
# MAGIC - This data fits in memory, so we hand the prepped table to **scikit-learn** via `toPandas()`. One **Pipeline** (`src/demo_utils/models.py`)
# MAGIC   imputes, scales/one-hot encodes (`handle_unknown="ignore"` covers categories only seen in test), then fits
# MAGIC - **Time-aware split**: train on the oldest 80% of shipments, validate on the newest 20%. Same idea as the hackathon's
# MAGIC   public/private leaderboard weeks: never let the model peek at the future
# MAGIC - Each model = one MLflow run with params, validation AUC, and the model + signature. Open **Experiments** on the right to compare.

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

mlflow.set_registry_uri("databricks-uc")
mlflow.sklearn.autolog(disable=True)

train_sdf = spark.table(T("silver_train"))
num_cols, cat_cols, dropped = split_feature_types(train_sdf, {ID, TARGET, DATE}, HIGH_CARDINALITY)
feature_cols = num_cols + cat_cols
print(f"numeric: {num_cols}\ncategorical: {cat_cols}\ndropped: {dropped}")

train_pdf = spark_prep(train_sdf, num_cols, cat_cols, keep=[TARGET, DATE]).toPandas().sort_values(DATE)
cut = int(len(train_pdf) * 0.8)
tr, va = train_pdf.iloc[:cut], train_pdf.iloc[cut:]
X_tr, y_tr, X_va, y_va = tr[feature_cols], tr[TARGET], va[feature_cols], va[TARGET]
print(f"train {len(X_tr):,} rows (to {tr[DATE].max()}) | validate {len(X_va):,} rows (from {va[DATE].min()})")

# COMMAND ----------

batch_id = uuid.uuid4().hex[:6]
results, pipes = [], {}
for name, (estimator, scale) in candidate_models("classification", SEED).items():
    pipe = Pipeline([("prep", preprocessor(num_cols, cat_cols, scale)), ("model", estimator)])
    with mlflow.start_run(run_name=f"{name}_{batch_id}") as run:
        pipe.fit(X_tr, y_tr)
        auc = score(pipe, X_va, y_va, "roc_auc", positive_label=1)
        mlflow.log_params({"model_type": name, "split": "time_aware_80_20", "n_train_rows": len(X_tr),
                           "n_numeric": len(num_cols), "n_categorical": len(cat_cols)})
        mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()})
        mlflow.log_metric("val_roc_auc", auc)
        # Serve probabilities so Spark batch scoring can rank shipments by risk
        signature = infer_signature(X_tr.head(100), pipe.predict_proba(X_tr.head(100)))
        info = mlflow.sklearn.log_model(pipe, name="model", signature=signature, input_example=X_tr.head(3),
                                        pyfunc_predict_fn="predict_proba")
    results.append({"model": name, "val_roc_auc": auc, "run_id": run.info.run_id, "model_uri": info.model_uri})
    pipes[name] = pipe
    print(f"{name:<24} val_roc_auc = {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pick the best run and register it in Unity Catalog
# MAGIC The winner becomes a versioned model at `workspace.default.shipment_delay_model`, with lineage, permissions and an `@champion` alias.
# MAGIC Re-run with a better model and it becomes the next version; anything loading `@champion` follows the alias.

# COMMAND ----------

from mlflow import MlflowClient

best = max(results, key=lambda r: r["val_roc_auc"])
board = pd.DataFrame(results)
board["role"] = np.where(board["model"] == best["model"], "🏆 champion", "challenger")
px.bar(board, x="model", y="val_roc_auc", color="role", text_auto=".4f",
       color_discrete_map={"🏆 champion": ORANGE, "challenger": "#9AA5AB"},
       title="Model leaderboard: validation ROC AUC (higher is better)").show()

mv = mlflow.register_model(best["model_uri"], MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Registered {MODEL_NAME} version {mv.version} ({best['model']}) as @champion")

exp_id = mlflow.get_run(best["run_id"]).info.experiment_id
displayHTML(links_html("🧪 Inspect it in the UI", {
    "MLflow experiment: compare all 3 runs side by side": f"{HOST}/ml/experiments/{exp_id}",
    f"Unity Catalog model: {MODEL_NAME} (versions, lineage, permissions)": f"{HOST}/explore/data/models/{MODEL_NAME.replace('.', '/')}",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why does the champion predict what it predicts?
# MAGIC **Permutation importance**: shuffle one column at a time and measure how much worse the model gets. Did the LLM news signal earn its place?
# MAGIC The chart is also logged to the champion's MLflow run.

# COMMAND ----------

from sklearn.inspection import permutation_importance

sample = X_va.sample(min(len(X_va), 3000), random_state=SEED)
imp = permutation_importance(pipes[best["model"]], sample, y_va.loc[sample.index], scoring="roc_auc",
                             n_repeats=3, random_state=SEED)
importance = (pd.DataFrame({"feature": sample.columns, "importance": imp.importances_mean})
                .sort_values("importance").tail(15))
fig = px.bar(importance, x="importance", y="feature", orientation="h", color="importance",
             color_continuous_scale="OrRd", title=f"What drives the {best['model']} champion? (drop in AUC when shuffled)")
fig.show()
with mlflow.start_run(run_id=best["run_id"]):
    mlflow.log_figure(fig, "feature_importance.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Batch scoring with Spark ➜ Gold
# MAGIC The registered model is loaded **by URI** (`models:/...@champion`) as a **Spark UDF**, so scoring runs distributed over the Delta table,
# MAGIC like a production batch job. Results land in `gold_predictions` (test features + delay probability + `risk_band`),
# MAGIC the table that powers Dashboards and Genie.

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, f"models:/{MODEL_NAME}@champion", env_manager="local", result_type="array<double>")
silver_test = spark.table(T("silver_test"))
p = F.col("_proba")[1]

preds = (spark_prep(silver_test, num_cols, cat_cols, keep=[ID])
         .withColumn("_proba", predict(*feature_cols))
         .select(ID, p.alias("delay_probability"),
                 F.when(p >= 0.7, "High").when(p >= 0.4, "Medium").otherwise("Low").alias("risk_band")))

gold = (silver_test.join(preds, on=ID, how="left")
        .withColumn("model_name", F.lit(MODEL_NAME))
        .withColumn("model_version", F.lit(str(mv.version)))
        .withColumn("scored_at", F.current_timestamp()))
gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_predictions"))
spark.sql(f"COMMENT ON TABLE {T('gold_predictions')} IS 'One row per test shipment with delay_probability (0-1) and risk_band "
          f"(High >= 0.7, Medium >= 0.4, else Low) from {MODEL_NAME} v{mv.version}. news_signal is the LLM label of that month''s regional news.'")

vis = spark.table(T("gold_predictions")).select("delay_probability", "risk_band").toPandas()
bands = vis["risk_band"].value_counts()
displayHTML(kpi_html({
    "Scored shipments": f"{len(vis):,}",
    "🔴 High risk": f"{bands.get('High', 0):,}",
    "🟠 Medium": f"{bands.get('Medium', 0):,}",
    "🟢 Low": f"{bands.get('Low', 0):,}",
    "Avg P(delay)": f"{vis['delay_probability'].mean():.1%}",
}))
px.histogram(vis, x="delay_probability", color="risk_band", nbins=40, color_discrete_map=RISK_COLORS,
             title="Where the test shipments land: predicted delay risk").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export a Kaggle-style `submission.csv` from Gold
# MAGIC A submission is just an export of the gold table in the exact format asked for. It's small, so we write **one** plain CSV with pandas
# MAGIC (Spark's `.write.csv` makes a folder of part files, which Kaggle rejects). Then **check the file before you submit**.

# COMMAND ----------

submission = (spark.table(T("gold_predictions"))
              .select(ID, F.col("delay_probability").alias(TARGET)).orderBy(ID).toPandas())
submission.to_csv(SUBMISSION_PATH, index=False)

expected = pd.read_csv(f"{VOLUME_PATH}/sample_submission.csv", dtype=str)
assert list(submission.columns) == list(expected.columns), "column names must match sample_submission"
assert len(submission) == len(expected), f"{len(submission)} rows, expected {len(expected)}"
assert set(submission[ID]) == set(expected[ID]), "IDs differ from sample_submission"
assert submission[TARGET].between(0, 1).all(), "probabilities must be in [0, 1]"
print(f"✅ {SUBMISSION_PATH}: {len(submission):,} rows, all checks passed")
display(submission.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. GenAI risk briefings with `ai_query` ✨
# MAGIC `ai_query()` calls a Foundation Model from SQL. Here an LLM writes an ops-ready briefing for the 5 riskiest shipments,
# MAGIC saved to `gold_ai_briefings`. A great way to explain your model's calls to judges.

# COMMAND ----------

context_cols = [c for c in spark.table(T("gold_predictions")).columns if c not in ("model_name", "model_version", "scored_at")]
q_ai = f"""
SELECT {ID}, delay_probability,
       ai_query('{AI_MODEL}', CONCAT(
         'You are a supply chain risk analyst. In at most 2 short sentences, explain what in this shipment drives its delay risk ',
         '(probability ', CAST(ROUND(delay_probability, 2) AS STRING), ') and give one concrete action to reduce it. ',
         'Shipment: ', to_json(struct(*)))) AS ai_briefing
FROM (SELECT {', '.join(context_cols)} FROM {T('gold_predictions')} ORDER BY delay_probability DESC LIMIT 5)
"""
try:
    spark.sql(q_ai).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_ai_briefings"))
    display(spark.table(T("gold_ai_briefings")))
except Exception as e:
    print(f"AI Functions unavailable right now ({str(e).splitlines()[0][:160]}). Skipping; everything else is done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Delta time travel ⏪
# MAGIC Every write is a new table version. Re-run the notebook and compare: `SELECT * FROM gold_predictions VERSION AS OF 0`.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {T('gold_predictions')}").select("version", "timestamp", "operation", "operationMetrics"))

# COMMAND ----------

displayHTML(links_html("🚀 Everything you just built (click to open)", {
    "🥇 gold_predictions: schema, sample data, lineage graph": f"{HOST}/explore/data/{T('gold_predictions').replace('.', '/')}",
    "📰 silver_news_signals: LLM labels for every headline": f"{HOST}/explore/data/{T('silver_news_signals').replace('.', '/')}",
    "✨ gold_ai_briefings: LLM-written risk notes": f"{HOST}/explore/data/{T('gold_ai_briefings').replace('.', '/')}",
    "🏛️ Registered model + @champion alias": f"{HOST}/explore/data/models/{MODEL_NAME.replace('.', '/')}",
    "🧪 MLflow experiment: all runs, metrics, artifacts": f"{HOST}/ml/experiments/{exp_id}",
    "📁 Volume: the CSVs and submission.csv": f"{HOST}/explore/data/volumes/{CATALOG}/{SCHEMA}/{VOLUME}",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genie & AI/BI Dashboard
# MAGIC
# MAGIC **Genie space** (New ➜ Genie space ➜ add `gold_predictions`, `silver_train`, `silver_news_signals`). Ask:
# MAGIC 1. *"How many test shipments are High risk, and what share of the total is that?"*
# MAGIC 2. *"Which origin regions have the highest average delay probability?"*
# MAGIC 3. *"Which news signals came before the most delays in silver_train?"*
# MAGIC
# MAGIC **AI/BI Dashboard** (New ➜ Dashboard ➜ dataset `gold_predictions`):
# MAGIC 1. **Bar chart**: average `delay_probability` by `origin_region`, colored by `news_signal`, as a "where to act first" view.
# MAGIC 2. **Counter + pie**: shipments per `risk_band`, plus a table of the 20 riskiest shipments for the ops team.
# MAGIC
# MAGIC ## 🏁 Apply it to your hackathon data
# MAGIC | In this demo | On your data (e.g. Supply Chain Disruptor) |
# MAGIC |---|---|
# MAGIC | 3 CSVs ➜ Bronze Delta tables | all 9 CSVs ➜ Bronze, one table each |
# MAGIC | `ai_classify` on headlines | `news.csv`: ~21 articles a week, and the signal is in the words (`ai_classify`, `ai_extract`, `ai_query`) |
# MAGIC | join news onto shipments by region + month | join weekly tables (prices, tariffs, shipping, macro) by week + material / country / lane |
# MAGIC | time-aware split | weeks 1–208 are history; validate on the latest weeks, never the future |
# MAGIC | classifier ➜ `delay_probability` | `disruption_prob` (calibration matters) |
# MAGIC | `models.py` also has regressors | `price_forecast` |
# MAGIC | submission checks | 260 rows, `w{week}_{material}` ids, shares sum to 1 |
# MAGIC
# MAGIC ### 🧰 Databricks toolkit (all on Free Edition; the track is 30% "use of Databricks")
# MAGIC | Tool | Use it to |
# MAGIC |---|---|
# MAGIC | **Databricks Assistant** (`Cmd/Ctrl + I`, or `/fix` on an error) | write, explain and debug code and SQL |
# MAGIC | **AI Functions** (`ai_classify`, `ai_extract`, `ai_query`, `ai_summarize`) | turn news text into features with one line of SQL |
# MAGIC | **Delta + Spark SQL** | store every file as a governed table, join and aggregate fast |
# MAGIC | **MLflow + Unity Catalog** | track every experiment and version every model |
# MAGIC | **Genie / AI/BI Dashboards** | show judges your results and let them ask questions |
# MAGIC | **Catalog Explorer ➜ Lineage** | show how data flows from CSV to submission |
# MAGIC | **Jobs + Asset Bundles** (see README) | re-run the whole pipeline when Phase 2 data drops Saturday at 12 PM |
