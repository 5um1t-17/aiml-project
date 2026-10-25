# Selected Kaggle dataset integration

Source: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
Kaggle reference: `architsharma01/loan-approval-prediction-dataset`; version 1; reported license MIT.

Raw CSV SHA-256: `4b5cd093d178378f4cfa8c107adb6e599b88be9d8a3b51f3b99c0d5914154e54`.

Rows: 4269; Approved: 2656; Rejected: 1613; inputs: 11.

## Units and provenance

Annual income follows `income_annum`. Monetary fields use full source values. Currency is not established. Loan term is interpreted as years by explicit project choice; the publisher metadata does not specify its unit. Collection institution and collected-versus-generated provenance are not established. These limitations appear in the website.

## Cleaning and training

Headers and string categories are stripped. No original missing cells or exact duplicate rows were found. Twenty-eight negative residential-asset values are marked missing and imputed using medians fitted within training folds. All rows and target labels are retained. Loan ID and the target are excluded from features.

New seed-42 stratified split: 3415 training / 854 held out. Five-fold training-only CV balanced accuracy selects parameters and the source-only or engineered variant. Engineered candidates include total assets, loan/income, assets/loan, principal/income under the years assumption, and log transforms.

## Final website training

After evaluation, the fixed selected pipelines are cloned and refitted on all 4,269 source rows. These final models use version `kaggle-archit-full-v2`. The separate evaluation models remain in `models/kaggle-evaluation-pipelines.joblib`, allowing the original unseen-holdout metrics to be reproduced. No independent test accuracy is claimed for the all-row refits. Training coverage checks use the full source ranges for new applications; old records retain their original scores and the corresponding evaluation-model support ranges. No new rows or invented labels are added. Smaller loans/incomes absent from the source remain unsupported.

## Held-out label-prediction results (evaluation models only)

| Model | Accuracy | Balanced accuracy | Rejected-label recall | ROC AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9450 | 0.9473 | 0.9567 | 0.9855 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9953 | 0.9938 | 0.9876 | 1.0000 |
| Equal-weight mean | 0.9988 | 0.9985 | 0.9969 | 1.0000 |

These scores describe this file's supplied labels and are not externally validated banking performance. The very high tree/ensemble results can reflect simple target patterns. Dataset-wide exploratory counts show a strong CIBIL/label association; that does not establish causality or justify a lending rule. Raw scores are not calibrated probabilities. Metrics are not directly comparable with results on the old mortgage dataset.

## Website/storage update

The form, API, sample upload CSV, full-data download, data-support checks, detail profile and exports use the new source-native fields. Existing mortgage records preserve old scores and schema; they are not passed to models requiring CIBIL/assets. SQLite migration retains IDs, predictions and staff-review history with a backup. New JSON inputs preserve full amounts and annual income. Human review remains separate from scoring.

## Reproduction

`python train.py` rebuilds final `models/kaggle-pipelines.joblib`, separate `models/kaggle-evaluation-pipelines.joblib` and `models/kaggle-metrics.json`. `python -m unittest test_project.py -v` verifies the local workflow. Raw source data and metadata are in `data/`.
