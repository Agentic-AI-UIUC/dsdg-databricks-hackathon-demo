# dsdg-databricks-hackathon-demo

An end-to-end demo of how to use Databricks on your hackathon data, made for the **DSDG Hackathon 2026** Databricks track.

It runs on a simple **sample dataset** (fake shipments plus regional news headlines) and shows every Databricks piece you can use on your own data:

**Volume ➜ Delta (Bronze/Silver/Gold) ➜ AI Functions on the news ➜ Spark SQL ➜ MLflow ➜ Unity Catalog ➜ Spark batch scoring ➜ Kaggle-style submission ➜ Genie & AI/BI Dashboards**

Built for [Databricks Free Edition](https://www.databricks.com/learn/free-edition) (serverless). A full run takes about 5 minutes.

## Repo layout

```
.
├── databricks.yml                     # Asset Bundle: what gets deployed
├── resources/end_to_end_demo.job.yml  # the serverless job that runs the notebook
├── notebooks/end_to_end_demo.py       # the demo, top to bottom
├── src/demo_utils/                    # helpers the notebook imports
│   ├── data.py          # CSV ➜ Spark with clean column names
│   ├── features.py      # Spark feature typing + prep
│   ├── models.py        # sklearn pipelines and metrics
│   ├── sample_data.py   # generates the sample dataset
│   └── viz.py           # banner, KPI tiles, link cards
├── pyproject.toml                     # dependencies; the notebook runs `%pip install ..`
├── uv.lock                            # pinned versions for local dev (`uv sync`)
└── .github/workflows/deploy.yml       # CD: push to main ➜ deployed to Databricks
```

## Run it: pick one

### 1. No install: Git folder in the Databricks UI
1. In Databricks: **Workspace ➜ Create ➜ Git folder**, paste this repo's URL.
2. Open `notebooks/end_to_end_demo` and click **Run all**.

### 2. Databricks CLI from your laptop
```bash
# install (macOS/Linux; on Windows: winget install Databricks.DatabricksCLI)
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks bundle deploy                 # uploads notebooks + src, creates the job
databricks bundle run end_to_end_demo    # runs it and streams the status
```
The job shows up under **Jobs & Pipelines** as `[dev <you>] end_to_end_demo`.

### 3. CD with GitHub Actions (deploy on every push)
1. Click **Use this template** to make your own copy of this repo.
2. In Databricks: **Settings ➜ Developer ➜ Access tokens ➜ Generate new token**.
3. In your GitHub repo: **Settings ➜ Secrets and variables ➜ Actions**, add
   - `DATABRICKS_HOST`: your workspace URL, e.g. `https://dbc-xxxx.cloud.databricks.com`
   - `DATABRICKS_TOKEN`: the token from step 2
4. Push to `main`. The **Deploy to Databricks** workflow validates and deploys the bundle.
   To also run the job, open **Actions ➜ Deploy to Databricks ➜ Run workflow** and choose `end_to_end_demo`.

## Local development (optional)

Code runs on Databricks, so locally you only need the CLI and, for editing, [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # local .venv with the same dependencies + ruff
uvx ruff check     # the lint step CI runs before every deploy
```

## Apply it to your hackathon data

The demo keeps its sample data in its own volume (`demo_sample`), so put your files in a separate one, then reuse the patterns from the notebook:

```bash
databricks volumes create workspace default hackathon MANAGED
databricks fs cp news.csv dbfs:/Volumes/workspace/default/hackathon/
```

| In this demo | On your data (e.g. Supply Chain Disruptor) |
|---|---|
| 3 CSVs ➜ Bronze Delta tables | all 9 CSVs ➜ Bronze, one table each |
| `ai_classify` on headlines | `news.csv`: the signal is in the words (`ai_classify`, `ai_extract`, `ai_query`) |
| join news onto shipments by region + month | join weekly tables by week + material / country / lane |
| time-aware split | validate on the latest weeks, never the future |
| classifier ➜ `delay_probability` | `disruption_prob` |
| regressors in `models.py` | `price_forecast` |
| submission checks | 260 rows, `w{week}_{material}` ids, shares sum to 1 |

## If it breaks

- **`FileNotFoundError` / volume errors**: the notebook creates `workspace.default.demo_sample` and writes the sample data itself. Check the `catalog` / `schema` / `volume` widgets at the top.
- **`ModuleNotFoundError: demo_utils`**: re-run the `%pip install ..` cell, and open the notebook from the Git folder or bundle so `pyproject.toml` and `src/` sit next to `notebooks/`.
- **MLflow / Unity Catalog register error**: re-run from the `%pip install` cell; the restart must happen before anything imports MLflow.
- **AI Functions step prints "unavailable"**: that's the built-in fallback, and the rest still runs. Try again in a minute or switch `AI_MODEL`.
- **Stuck or slow**: Run ➜ Clear state, or detach and reattach serverless compute, then **Run all**.
