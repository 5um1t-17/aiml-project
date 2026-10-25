# Model retraining audit

All three algorithm families were rebuilt from the unchanged source CSV for each dataset option. Target labels and stored applications were not used to manufacture new training examples.

## Asset-based

Completed: 25 October 2026, 09:01:45 PM IST

Source rows: 4,269. Evaluation training/test: 3,415/854. Final website fitting: all 4,269 rows.

| Algorithm | Held-out accuracy | Balanced accuracy |
|---|---:|---:|
| Logistic Regression | 94.50% | 94.73% |
| Decision Tree | 100.00% | 100.00% |
| Random Forest | 99.53% | 99.38% |
| Equal-weight mean | 99.88% | 99.85% |

Checks passed: complete disjoint train/test partition; no identical-input group crossing holdout; ID and target excluded; both labels represented; convergence warnings treated as errors; actual evaluation/final scaler fitted row counts verified.

The scores belong to separate evaluation models. The final website models were fitted on all source rows after evaluation and have no separate unseen-test accuracy measurement.

Source CSV SHA-256: `4b5cd093d178378f4cfa8c107adb6e599b88be9d8a3b51f3b99c0d5914154e54`

Environment: `{"joblib": "1.6.0", "numpy": "2.5.4", "pandas": "3.1.0", "python": "3.14.3", "scikit_learn": "1.9.1"}`

- `models/kaggle-pipelines.joblib`: `f0a6d7af1dd15e5edaea235b2ffa6ee9ce036438a677d88b0d2a341e18d9b98b`
- `models/kaggle-evaluation-pipelines.joblib`: `7dea194b324126003d8e5e0e86005a4d61d1ff2a2538fc9a3dff281a828d01bb`

## Small / personal loans

Completed: 25 October 2026, 09:01:54 PM IST

Source rows: 45,000. Evaluation training/test: 36,000/9,000. Final website fitting: all 45,000 rows.

| Algorithm | Held-out accuracy | Balanced accuracy |
|---|---:|---:|
| Logistic Regression | 84.06% | 87.46% |
| Decision Tree | 88.30% | 88.09% |
| Random Forest | 89.07% | 90.11% |
| Equal-weight mean | 87.26% | 89.41% |

Checks passed: complete disjoint train/test partition; no identical-input group crossing holdout; ID and target excluded; both labels represented; convergence warnings treated as errors; actual evaluation/final scaler fitted row counts verified.

The scores belong to separate evaluation models. The final website models were fitted on all source rows after evaluation and have no separate unseen-test accuracy measurement.

Source CSV SHA-256: `67cca604cd71cbd605014f721d078d20d44d726d625945591450733535f9868a`

Environment: `{"joblib": "1.6.0", "numpy": "2.5.4", "pandas": "3.1.0", "python": "3.14.3", "scikit_learn": "1.9.1"}`

- `models/personal-pipelines.joblib`: `5c88fbd446674ef6f3fd663810c4f92df962d0a6a2d875935a35a85443a29d0b`
- `models/personal-evaluation-pipelines.joblib`: `25e3459850009da4d16e496caa020059d49b0e9b6039091d17eb6a28f520ed54`

## Scope and history

The sources remain separate. The personal-loan source is synthetic and was expanded by its publisher before publication; nearby records can make internal evaluation optimistic. Currency is not established, asset-source term years remains a project assumption, and neither source establishes real-bank accuracy or calibrated approval probabilities.

Existing inputs, prediction snapshots and staff reviews were checked against a before-training database hash and are unchanged. Previous model artifacts were archived before training.

## Verification

All 43 tests passed, including saved-model holdout reproduction, all-row fit counts, split-leakage guards, source-unit validation, both prediction APIs, history/export integrity, batch atomicity and staff-review protections.
