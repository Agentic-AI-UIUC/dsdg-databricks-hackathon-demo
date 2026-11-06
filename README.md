# dsdg-databricks-hackathon-demo

A bare-bones tour of Databricks for the **DSDG Hackathon 2026** Databricks track. One short cell per tool, on Databricks' built-in `samples` tables:

**Volumes ➜ Delta tables ➜ SQL ➜ Time travel ➜ AI Functions ➜ MLflow + Unity Catalog ➜ Export to CSV ➜ Genie & Dashboards**

It doesn't touch the competition data; use the same tools on your own files. Runs on [Databricks Free Edition](https://www.databricks.com/learn/free-edition) (serverless) in about 2 minutes.

## What's in the repo

```
.
├── notebooks/intro_to_databricks.py      # the tour
├── databricks.yml                        # Asset Bundle: what gets deployed
├── resources/intro_to_databricks.job.yml # serverless job that runs the notebook
├── pyproject.toml                        # dependencies; the notebook runs `%pip install ..`
└── .github/workflows/deploy.yml          # CD: push to main ➜ deployed to Databricks
```

## Run it: pick one

### 1. No install: Git folder in the Databricks UI
1. **Workspace ➜ Create ➜ Git folder**, paste this repo's URL.
2. Open `notebooks/intro_to_databricks` and click **Run all**.

### 2. Databricks CLI from your laptop
```bash
# install (macOS/Linux; on Windows: winget install Databricks.DatabricksCLI)
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks bundle deploy                    # uploads the notebook, creates the job
databricks bundle run intro_to_databricks   # runs it
```

### 3. CD with GitHub Actions (deploy on every push)

> [!IMPORTANT]
> The workflow can't reach your workspace until you add **two repository secrets**. Without them the
> **Deploy to Databricks** run fails at "Check secrets" (a red ❌ on a fresh copy is expected until you do this).

1. Click **Use this template** to make your own copy of this repo.
2. In Databricks: **Settings ➜ Developer ➜ Access tokens ➜ Generate new token**. Copy it; it's shown once.
3. In your GitHub repo: **Settings ➜ Secrets and variables ➜ Actions ➜ New repository secret**, and add both:

   | Name | Value |
   |---|---|
   | `DATABRICKS_HOST` | your workspace URL, e.g. `https://dbc-xxxx.cloud.databricks.com` |
   | `DATABRICKS_TOKEN` | the token from step 2 |

   Treat the token like a password: never paste it into code, a notebook, or a commit.
4. Push to `main` (or **Actions ➜ Deploy to Databricks ➜ Run workflow**).

## Local development (optional)

```bash
uv sync            # local .venv with the same dependencies + ruff
uvx ruff check     # the lint step CI runs before every deploy
```

## If it breaks

- **AI Functions cell errors**: re-run it in a minute; the Foundation Model endpoint can be briefly busy.
- **MLflow register error**: re-run from the `%pip install` cell; the restart must happen before anything imports MLflow.
- **Stuck or slow**: Run ➜ Clear state, or detach and reattach serverless compute, then **Run all**.
