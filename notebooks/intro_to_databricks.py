# Databricks notebook source
# MAGIC %md
# MAGIC # Intro to Databricks: a quick tool tour
# MAGIC
# MAGIC One short cell per tool, using Databricks' built-in `samples` tables (a bakery chain and NYC taxi trips).
# MAGIC Nothing here is about the hackathon; swap in your own files and tables when you're ready.
# MAGIC
# MAGIC **Volumes ➜ Delta tables ➜ SQL ➜ Time travel ➜ AI Functions ➜ MLflow + Unity Catalog ➜ Export to CSV ➜ Genie & Dashboards**

# COMMAND ----------

# MAGIC %pip install -q ..

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG, SCHEMA, VOLUME = "workspace", "default", "intro_demo"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Volumes: where your files live
# MAGIC A Volume is a governed folder for raw files (CSV, JSON, images). Upload yours with
# MAGIC **Catalog ➜ workspace ➜ default ➜ Volumes ➜ your volume ➜ Upload to this volume**.
# MAGIC Here we drop a sample CSV in so there's something to load.

# COMMAND ----------

spark.table("samples.bakehouse.sales_franchises").toPandas().to_csv(f"{VOLUME_PATH}/franchises.csv", index=False)
display(dbutils.fs.ls(VOLUME_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. CSV ➜ Delta table
# MAGIC Read the file with Spark and save it as a **Delta table**: queryable with SQL, versioned, and visible in Catalog.

# COMMAND ----------

df = spark.read.option("header", True).option("inferSchema", True).csv(f"{VOLUME_PATH}/franchises.csv")
df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.franchises")
display(spark.table(f"{CATALOG}.{SCHEMA}.franchises"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SQL on your tables
# MAGIC SQL cells work right in the notebook. Click **+ ➜ Visualization** on the result to turn it into a chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT country, COUNT(*) AS franchises
# MAGIC FROM workspace.default.franchises
# MAGIC GROUP BY country
# MAGIC ORDER BY franchises DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Time travel
# MAGIC Every write to a Delta table is a new version, so you can see history and query old versions (`VERSION AS OF 0`).

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY workspace.default.franchises

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. AI Functions: LLMs inside SQL
# MAGIC No API keys and no extra libraries. Try them on any text column.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   review,
# MAGIC   ai_analyze_sentiment(review) AS sentiment,
# MAGIC   ai_classify(review, ARRAY('product', 'service', 'price', 'location')) AS topic,
# MAGIC   ai_summarize(review, 15) AS summary
# MAGIC FROM samples.bakehouse.media_customer_reviews
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow + Unity Catalog: track and register a model
# MAGIC A deliberately tiny model (taxi fare from trip distance). MLflow records the run; Unity Catalog versions the model.
# MAGIC Open **Experiments** in the sidebar to see the run, and **Catalog ➜ Models** to see the registered model.

# COMMAND ----------

import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

mlflow.set_registry_uri("databricks-uc")

trips = spark.table("samples.nyctaxi.trips").select("trip_distance", "fare_amount").limit(5000).toPandas()
X_train, X_test, y_train, y_test = train_test_split(trips[["trip_distance"]], trips["fare_amount"], random_state=0)

with mlflow.start_run(run_name="taxi_fare_baseline"):
    model = LinearRegression().fit(X_train, y_train)
    mlflow.log_metric("mae", mean_absolute_error(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                             registered_model_name=f"{CATALOG}.{SCHEMA}.taxi_fare_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Export to CSV
# MAGIC Write a query result to one CSV in your Volume, then download it from **Catalog ➜ Volumes ➜ intro_demo ➜ ⋮ ➜ Download**.
# MAGIC (Use pandas `to_csv`: Spark's `.write.csv` makes a folder of part files.)
# MAGIC No code? Any result table in a notebook or the SQL Editor has a **⬇ Download** button.

# COMMAND ----------

summary = spark.sql("SELECT country, COUNT(*) AS franchises FROM workspace.default.franchises GROUP BY country")
summary.toPandas().to_csv(f"{VOLUME_PATH}/franchises_by_country.csv", index=False)
display(dbutils.fs.ls(VOLUME_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Genie & Dashboards (no code)
# MAGIC - **Genie**: New ➜ Genie space ➜ add `workspace.default.franchises` ➜ ask *"Which country has the most franchises?"*
# MAGIC - **Dashboard**: New ➜ Dashboard ➜ add a dataset from any table ➜ drag in a chart.
# MAGIC - **Databricks Assistant**: press `Cmd/Ctrl + I` in any cell to write or explain code, or type `/fix` on an error.
# MAGIC
# MAGIC **Your turn:** upload your own CSVs to a Volume and use these same tools on them.
