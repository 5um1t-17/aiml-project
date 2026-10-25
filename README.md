# LoanLens — Two-dataset Kaggle loan assessment website

A local semester AIML website with Logistic Regression, Decision Tree and Random Forest, an equal-weight model-score average, CSV batch analysis, SQLite history and explicit staff review.

## Run locally on Windows

Double-click `start.bat`, or use PowerShell:

```powershell
cd "C:\Users\sumit\Downloads\aiml project"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe train_personal.py
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 . Existing `.venv` and model artifacts are already supplied. Training is needed only when intentionally rebuilding the models. Exact installed dependency versions are in `requirements-lock.txt`.

## Smaller loans and incomes

The default `/predict` page now uses a **separate** small-loan dataset from https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data . The publisher identifies its 45,000 records as synthetic, expanded with SMOTENC before publication (Apache 2.0 reported by Kaggle). Full source loan amounts range from 500 to 35,000; annual incomes start at 8,000. Currency is not specified, and values must not be assumed equivalent to the asset-based dataset or to rupees/dollars. Source labels are retained: publisher mapping 1 = approved, 0 = rejected.

Nine financial/credit inputs are used. Age, gender, education and the supplied derived income percentage are excluded. Impossible employment experience over 80 years is treated as missing and imputed within folds. The source has no asset values or term. User-entered term in years is saved only for principal arithmetic and is never passed to these classifiers. No missing source fields or labels are fabricated.

The three model families have predeclared parameters, three-fold grouped CV, and a separate grouped 80/20 evaluation. Identical selected-feature groups stay in one split. Upstream synthetic near-neighbours can still inflate evaluation, so this is an educational demo rather than external banking validation. Final pipelines are cloned and refitted on all 45,000 rows. Metrics apply to separate evaluation models; no independent test score is claimed for the final full-data fits.

- `/predict?dataset=personal`, `/comparison?dataset=personal`: small-loan form and evaluation.
- `/predict?dataset=assets`, `/comparison?dataset=assets`: original asset-based models.
- `/batch/personal`, `/sample/personal.csv`: matching small-loan upload and sample.
- `/dataset/personal.csv`: original unchanged public source file.
- `personal_loans.py`, `personal_routes.py`, `train_personal.py`: source schema, routes and reproducible training.
- `data/personal_dataset_source.json`: source metadata, hash and provenance.
- `models/personal-pipelines.joblib`, `models/personal-evaluation-pipelines.joblib`, `models/personal-metrics.json`: final models, evaluation models and results.

History, staff review, print reports and exports support both schemas. Existing scores are retained. Inputs outside either source still show review flags; using more data does not cover every possible amount.

## Asset-based dataset and units

Source: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

`data/kaggle_loan_approval.csv` is the unmodified downloaded source: 4,269 rows and 13 columns. Kaggle API metadata reports MIT licensing, version 1. The source metadata and checksum are recorded in `data/dataset_source.json`. The metadata does not establish the collecting institution, generation process, currency or term unit.

- Income is **annual** (`income_annum`), not monthly.
- Loan and asset amounts are **full source values**, without division by 1,000 for inference.
- Loan term is interpreted as **years by explicit user-selected project assumption**, since publisher metadata does not specify its unit. The website states this assumption.
- CIBIL score is entered directly on the source scale, 300–900.
- Four asset-value fields, dependents, education and self-employment are collected.
- Co-applicant income, property area, binary credit-history criteria, age and gender are not collected because they are not source model features.

The original headers/category values contain extra spaces, which are stripped in memory. The 28 negative residential-asset values are treated as missing and imputed within training folds. The raw file, source rows and labels are unchanged. No ID or target label enters model features.

## ML workflow

A new stratified 80/20 split uses 3,415 training and 854 held-out records with seed 42. Median imputation, scaling, encoding and derived financial ratios are fitted within training folds. A small grid search compares source-feature baselines with ratio-feature candidates using five-fold CV balanced accuracy. Choices are made using training data only. Individual model and equal-weight-average metrics are shown in Model performance. These belong to evaluation models saved as `models/kaggle-evaluation-pipelines.joblib`. After evaluation, fixed selected pipelines are cloned and refitted on **all 4,269 source rows** for website inference, saved as `models/kaggle-pipelines.joblib` with version `kaggle-archit-full-v2`. Final models have no independent unseen-test measurement; their training coverage is computed from all source rows. Existing saved applications retain their original model version and scores. Using all rows does not add smaller amounts missing from the source.

See `docs/KAGGLE_DATASET.md` for the actual scores. These measure provided label prediction on this file, not external bank accuracy; high scores can reflect simple label patterns. The percentages are uncalibrated model scores. Metrics on this dataset cannot be compared as an improvement over the different old 614-row dataset.

Each retraining run now rejects row overlap, identical-input leakage across the holdout, missing/invalid binary labels, ID/target inputs and convergence failures. The `training_run` section of each metrics JSON records completion time (UTC), package versions, label counts, split checks and actual fitted row counts for evaluation/final pipelines. The latest verified rebuild is documented in `docs/RETRAINING_AUDIT.md`.

## Website workflow

- `/`: overview, staff decision counts, latest records and model performance.
- `/predict`: source-matched annual-income/CIBIL/assets form and model assessment.
- `/batch`: upload up to 200 UTF-8 CSV rows (1 MB); `/sample.csv` supplies four fictional examples with exact headers. Extra ID/label columns are ignored. Every row is checked before an atomic batch save.
- `/history`: search by ID, education, demo label (legacy area searches also remain possible), status filters and CSV export.
- `/applications/<id>`: saved source inputs, raw scores, support warnings, linear-model explanation, staff review and a printable report.
- `/comparison`: individual and mean-model metrics, confusion matrices and training coverage.
- `/about`: source, units, methods and limitations; `/dataset.csv` downloads the unmodified training file.

Repayment checks use annual income / 12 and loan amount / (term-years × 12), excluding interest, existing debts and living costs. Out-of-range inputs and sparse durations are review prompts. Passing the checks does not certify reliability. A flagged case shows **Manual review required**, with raw scores under an expandable section. Input changes clear old displayed assessments.

Every application starts pending. Staff explicitly record Accept, Reject or Keep pending with notes, a CSRF-protected form and a stale-review check. Reviewer names and Employee/Manager roles are self-entered local-demo fields, not authenticated accounts or verified bank permissions. The models do not record staff decisions.

## PostgreSQL connection and migration

Copy `.env.example` to `.env` and set `DATABASE_URL` to the complete PostgreSQL URI, preserving its SSL parameters. Set stable `SECRET_KEY` and `BATCH_RECEIPT_SECRET` values. `.env` and database backups are excluded from Git. `DATABASE_SCHEMA` defaults to `public`.

```powershell
.\.venv\Scripts\python.exe check_postgres.py
.\.venv\Scripts\python.exe migrate_postgres.py
.\.venv\Scripts\python.exe app.py
```

The migration backs up local SQLite first, creates PostgreSQL tables, copies original applications/predictions/reviews without recomputing scores, verifies every field and synchronizes identity sequences. It runs in a transaction and refuses to overwrite a destination containing different records. Running it again against an identical destination is a no-op. Run the migration during a brief maintenance stop so the local snapshot includes all committed records before switching the website.

When DATABASE_URL is set, the app uses PostgreSQL. PostgreSQL schema creation/migration is an explicit maintenance command; server startup never seeds or resets the remote database. Without DATABASE_URL, local development retains SQLite compatibility. Vercel requires DATABASE_URL and a stable SECRET_KEY. Batch receipt signing uses BATCH_RECEIPT_SECRET or the same stable application secret.

Queries remain parameterized. PostgreSQL uses double precision for saved scores, identity IDs, foreign keys, atomic batch transactions and transaction-scoped advisory locks for stale-review protection. Errors returned/logged by the app do not include driver connection details. The SQLite-only reset helper refuses a PostgreSQL target; back up remote data with the database provider before maintenance.

`check_postgres.py` verifies the connection and table counts without printing credentials. PostgreSQL integration tests run only with explicit opt-in and use a disposable, uniquely named schema:

```powershell
$env:RUN_POSTGRES_TESTS="1"
.\.venv\Scripts\python.exe -m unittest test_postgres -v
Remove-Item Env:RUN_POSTGRES_TESTS
```

They verify both prediction APIs, filters, exports, foreign keys, batch rollback, concurrent stale-review handling, migration idempotency, conflict refusal and generated IDs. They never reset application tables in the configured production schema.

## Database and legacy history

`data/loanlens.db` is the offline SQLite database. With DATABASE_URL configured, PostgreSQL is the active persistent database. Canonical new inputs are saved as JSON with schema `kaggle-archit-v1` and a model version. Existing mortgage-format inputs and scores are preserved under their old schema. CIBIL scores, asset values and new-model scores are never inferred for old records. Uncollected old fields are NULL for new records; derived legacy columns are used only for shared table display.

Schema upgrades create a backup under `data/backups/`. The four fictional DEMO defaults are separate from training data: two personal-loan cases and two asset-based cases, with plausible within-source amounts. They cover an approval indication, a decline indication, a pending asset-backed review and a principal-repayment warning. Demo Employee/Manager reviews explicitly marked SIMULATED DEMO REVIEW illustrate accepted/rejected staff workflows; the remaining two are pending. Their scores are computed by the real trained pipelines. Empty-workspace startup seeds these four once and never resets populated history. A source update can replace only obsolete unreviewed demo fixtures after backup; reviewed examples and all non-demo applications remain intact.

CSV exports include canonical source fields, schema/model version, mean scores, support flags and latest staff review. Old-format values are exported separately with a legacy JSON snapshot. Text is escaped against spreadsheet formulas.

## Important files

- `loan_schema.py`: source-native input names and labels.
- `train.py`, `features.py`, `prediction.py`: training, derived features, validation, inference and diagnostics.
- `models/kaggle-pipelines.joblib`: final website models fitted on all 4,269 rows.
- `models/kaggle-evaluation-pipelines.joblib`: separate models fitted on 3,415 rows for held-out evaluation.
- `models/kaggle-metrics.json`: evaluation results, final-model training coverage and artifact versions.
- `models/pipelines-before-kaggle.joblib`, `models/metrics-before-kaggle.json`: previous model snapshots.
- `docs/archive/mortgage-v2/`: historical source/report references.
- `database.py`, `demo_data.py`: persistence, schema migration and demo fixtures.

Never load joblib models from untrusted sources. The supplied artifacts were trained locally from the documented CSV.

## Verify

```powershell
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
```

Tests use isolated temporary databases and fictional inputs. They cover units, schema migration, CSV handling, model averaging, explanations, staff review, history and recoverable demo resets.

## Hosting status

This version remains local. GitHub can store the code and source/model files. PostgreSQL and environment-based session configuration are implemented. Vercel deployment still needs runtime/package pinning, static-asset packaging, bundle exclusions and appropriate access protection. Configure DATABASE_URL, SECRET_KEY and BATCH_RECEIPT_SECRET in Vercel before deployment; local SQLite is not persistent Vercel storage. Do not commit `.venv`, `.env`, database files or backups; `.gitignore` excludes them.
