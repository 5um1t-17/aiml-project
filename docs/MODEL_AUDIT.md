> Historical mortgage-model audit. The active dataset and current metrics are documented in KAGGLE_DATASET.md.

# LoanLens data and model audit

## Why the reported example failed

The reported input was monthly applicant income 500, co-applicant income 148, full loan amount 20,000, and a 12-month term. Combined monthly income is 648. Principal-only repayment is 20,000 / 12 = 1,666.67 per month, or 257.2% of reported income, before interest and expenses. This is a repayment-capacity review flag, not an automatic lending rejection.

Only one of the 491 training records uses a 12-month term. 409 use 360 months. Combined income in the training split ranges from 1,442 to 81,000. The example is outside this income range. The old site displayed an extrapolated average as an approval indication.

## Dataset findings

- Rows: 614; approved: 422; declined: 192.
- Exact duplicate rows: 0. Conflicting labels for identical feature groups: 0.
- Missing selected-feature cells: 133.
- CSV SHA-256: `3ac62ced374efaf4581a185a2456ed8288081e934d0841ad170f50a1d23ff835`.
- Original labels and records are retained. No label was changed to force the example to decline.

The matching original practice challenge describes a mortgage approval problem: https://www.analyticsvidhya.com/datahack/contest/practice-problem-loan-prediction-2/ . The downloaded public mirror remains documented in README.md. Currency and institutional provenance are not established here. This is not a representative all-purpose bank lending dataset.

## Changes

1. Added total income, loan / annual income, monthly principal / income, log total income and log loan amount inside the fitted pipeline. Missing-value medians used for these features are learned only from each training fold.
2. Compared the original pipelines against a small training-only grid search for all three syllabus algorithms. Selection uses balanced accuracy to account for both approval and decline detection. Revised pipelines are retained only if their CV balanced accuracy improves.
3. Evaluated the equal-weight average on the same fixed holdout. The holdout was reused from earlier versions and is not fresh external validation. No model choice uses holdout metrics.
4. Added arithmetic and training-support checks. Flagged cases show Manual review required; raw scores remain inspectable without an approval endorsement. Checks do not record final acceptance/rejection.
5. Preserved legacy saved scores and tagged new predictions with model version. Historical scores are not silently recalculated.
6. Removed stale displayed results when inputs change.

## Fixed-holdout results

| Metric | Previous average | Audited average |
|---|---:|---:|
| Accuracy | 0.8537 | 0.8943 |
| Balanced accuracy | 0.7704 | 0.8435 |
| Decline recall | 0.5526 | 0.7105 |
| ROC AUC | 0.8084 | 0.8765 |
| Brier loss (lower is better) | 0.1265 | 0.1203 |

Individual holdout accuracy can decrease while decline detection improves. Cross-validation selection estimates have tuning optimism. Passing diagnostic checks is not proof that the models are reliable. The raw scores are uncalibrated and neither the models nor the arithmetic checks are validated bank underwriting policy.

## Reproduce

Run `.venv\Scripts\python.exe train.py` for training and `.venv\Scripts\python.exe -m unittest test_project.py -v` for regression tests. `models/metrics.json` contains numerical results, support ranges and model-selection metadata. Pre-audit artifacts are preserved in `models/pipelines-before-audit.joblib` and `models/metrics-before-audit.json`.

## Method references

- scikit-learn cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html
