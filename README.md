# LoanLens

An educational loan-review website using Logistic Regression, Decision Tree and Random Forest. It provides two dataset options, an equal-weight average model score, application history, CSV processing and a separate staff-review workflow.

## Features

- Small/personal-loan and asset-based-loan forms with source-specific inputs.
- Three trained classifiers for each dataset option.
- Model performance tables, confusion matrices and ROC graphs.
- Amounts displayed in words using thousand, lakh and crore wording.
- Repayment arithmetic and training-range review flags.
- PostgreSQL storage, with SQLite available for offline development.
- Application search, status filters, CSV upload/export and printable reports.
- Staff acceptance/rejection with review notes and an audit history.

## Datasets

| Option | Source | Records | Evaluation split |
|---|---|---:|---|
| Small/personal loans | [Taweilo: Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) | 45,000 | 36,000 training / 9,000 held out |
| Asset-based loans | [Archit Sharma: Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) | 4,269 | 3,415 training / 854 held out |

The datasets are kept separate because their fields and units differ. The small-loan publisher identifies its dataset as synthetic and expanded with SMOTENC before publication. Its loan amounts range from 500 to 35,000, with annual incomes from 8,000 in source units.

Income is entered as a full annual amount. Loan and asset amounts use full source values. Neither source establishes a currency, so the website does not imply a rupee/dollar conversion.

The asset-based dataset's term values are interpreted as years by project assumption. The small-loan dataset has no term column: the term entered on that form is used only for principal repayment arithmetic and is excluded from the classifiers. Its general credit score is not identified as CIBIL.

## Run locally on Windows

Use Python 3.14 for the environment matching the supplied model artifacts. Open PowerShell in the project folder containing `app.py` and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The repository includes the trained models; startup performs inference without retraining. Configure the database as described below, then start the website:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Database configuration

For a fresh setup, create a local `.env` from the supplied template if one does not already exist:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Configure these environment variables locally in `.env` or through the hosting platform:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Complete PostgreSQL connection URI, including its SSL settings. Leave empty for offline SQLite development. |
| `DATABASE_SCHEMA` | PostgreSQL schema name; defaults to `public`. |
| `SECRET_KEY` | A stable, private Flask session-signing key. Required on Vercel. |
| `BATCH_RECEIPT_SECRET` | A stable, private signing key for batch-upload receipts; defaults to `SECRET_KEY`. |

For a new PostgreSQL database, initialize the application tables once before starting the site:

```powershell
.\.venv\Scripts\python.exe -c "from configuration import database_target; from database import init_database; init_database(database_target())"
```

The PostgreSQL database stores applications, saved model scores and staff reviews. The application does not reset or seed PostgreSQL records during startup. Offline SQLite development creates its local database automatically and seeds four fictional demos only when the workspace is empty.

Keep `.env`, local database files and backups out of Git. Only `.env.example`, containing empty settings, belongs in the repository.

## Model evaluation

The supplied models were trained with source header/category whitespace cleanup, invalid-value handling and preprocessing fitted within training folds. Loan IDs and target labels are excluded from model inputs. Age, gender and education are excluded from the small-loan classifiers.

The asset-based workflow uses training-only five-fold cross-validation for selection. The small-loan workflow uses predeclared model settings and three-fold grouped cross-validation; identical selected-feature groups stay in one split. The publisher's upstream synthetic generation may still make its internal evaluation optimistic.

Displayed accuracy, confusion matrices and ROC curves come from held-out evaluation performed before final refitting. Each final website model is then fitted on every row of its own source dataset. Those final all-row models have no separate unseen-test measurement. Scores are uncalibrated and do not establish real-bank approval probabilities.

This repository contains the website runtime, trained prediction pipelines and saved evaluation metrics. Training scripts, automated tests, evaluation pipeline copies and development reports are maintained outside the runtime repository. The website loads the supplied artifacts directly and reports a clear startup error if one is missing.

## Main pages

| Route | Purpose |
|---|---|
| `/` | Overview and application activity |
| `/predict?dataset=personal` | Small/personal-loan check |
| `/predict?dataset=assets` | Asset-based-loan check |
| `/comparison?dataset=personal` | Small-loan model metrics, matrices and ROC graphs |
| `/comparison?dataset=assets` | Asset-based model metrics, matrices and ROC graphs |
| `/batch/personal` | Small-loan CSV upload |
| `/batch` | Asset-based CSV upload |
| `/history` | Saved applications, filters and exports |
| `/applications/<id>` | Application details, staff reviews and printable report |
| `/about` | Source information, methods and limitations |
| `/health` | Service status and configured database backend |

Sample CSVs are available at `/sample/personal.csv` and `/sample.csv`. The original source files can be downloaded from `/dataset/personal.csv` and `/dataset.csv`.

## Project structure

```text
app.py                 Flask application and routes
configuration.py       Environment configuration
postgres_backend.py    PostgreSQL connection support
database.py            Persistence and transaction handling
personal_routes.py     Small-loan prediction and CSV routes
personal_loans.py      Small-loan schema and diagnostics
loan_schema.py         Asset-based input schema
prediction.py          Inference, averaging and explanations
features.py            Pipeline feature transformers
amounts.py             Amount-to-words formatting
demo_data.py           Fictional sample fixtures
templates/             Website pages
static/                Styles and JavaScript
data/                  Two source CSVs offered for download
models/                Final fitted pipelines and metrics
requirements.txt       Runtime dependency installation entrypoint
requirements-lock.txt  Exact tested dependency versions
.env.example           Empty environment template
.gitignore             Runtime allowlist and local-file exclusions
.gitattributes         Dataset/model byte preservation
```

The models directory contains exactly these runtime files:

```text
models/kaggle-pipelines.joblib
models/kaggle-metrics.json
models/personal-pipelines.joblib
models/personal-metrics.json
```

The data directory contains only `kaggle_loan_approval.csv` and `personal_loan_data.csv`. Source provenance, preprocessing notes and evaluation results are embedded in the metrics JSON files. Every tracked template and static asset is used by the website. Personal settings, local databases, backups and development files are excluded from Git.

## Staff review and limitations

The mean score is the arithmetic average of the three model approval scores. A flagged case shows **Manual review required**. Principal-only repayment checks exclude interest, existing debts and living costs; passing these checks does not certify eligibility.

New applications start pending. A staff reviewer explicitly records acceptance, rejection or continued review with notes. Demo Employee/Manager entries are clearly identified as simulated examples. Reviewer names and roles are self-entered demo fields rather than authenticated staff accounts.

This project is intended for educational demonstrations. Institutional use requires suitable data, independent validation, calibration and appropriate authentication/access controls.

## Hosting

PostgreSQL and environment-based signing keys are supported. Vercel deployment additionally requires runtime/dependency compatibility, static-asset packaging, bundle exclusions and appropriate access protection. Set the database connection and signing keys through the hosting platform's environment settings.
