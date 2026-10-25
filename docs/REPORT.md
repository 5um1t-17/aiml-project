# Experiment 10 — AIML Mini Project
## LoanLens: Loan Assessment with Three Classification Algorithms

**Students / roll numbers:** ____________________  
**Class / division:** ____________________  
**College / guide:** ____________________

### Objective
Build a functional website that takes source-matched financial inputs, compares Logistic Regression, Decision Tree and Random Forest, reports their equal-weight mean score, and records an independent staff decision. The application illustrates supervised learning, preprocessing, feature engineering, validation, evaluation and database integration.

### Dataset and preprocessing
The selected Kaggle Loan-Approval-Prediction-Dataset by architsharma01 contains 4,269 rows with 2,656 Approved and 1,613 Rejected labels. Eleven input fields cover annual income, loan amount, loan term, CIBIL score, dependents, education, self-employment and four asset values. Loan ID and loan_status are excluded from features.

Source header/category whitespace is stripped. There are no original missing cells or exact duplicate rows. Twenty-eight negative residential-asset values are treated as missing; medians are learned within each training fold. Numeric fields are scaled and categories are one-hot encoded. Candidate features include total assets, loan/income, assets/loan, principal/income and logs. The raw CSV and target labels are unchanged.

Income is annual and monetary values are entered in full source units. Currency is not documented. Term-years is an explicit project assumption accepted by the user because publisher metadata does not establish the unit. The website discloses it, and repayment estimates use this assumption consistently.

### Training and evaluation
A new stratified seed-42 split uses 3,415 training and 854 held-out rows. Five-fold training-only CV balanced accuracy chooses parameters and whether derived features improve the baseline. The holdout is excluded from fitting and selection of the evaluation models. After evaluation, fixed selected pipelines are cloned and fitted on all 4,269 rows for website inference; the displayed test metrics apply only to the separate evaluation models. The website reports accuracy, balanced accuracy, precision, recall, F1, rejected-label recall, ROC AUC, Brier loss, confusion matrices and ROC curves for individual models and/or their equal-weight mean. CV values include model-selection optimism.

| Model | Holdout accuracy | Balanced accuracy |
|---|---:|---:|
| Logistic Regression | 94.50% | 94.73% |
| Decision Tree | 100.00% | 100.00% |
| Random Forest | 99.53% | 99.38% |
| Equal-weight mean | 99.88% | 99.85% |

These reproduce supplied labels on this file. They do not establish real bank accuracy, fairness, calibration or causal explanations. Source collection/generation provenance is not established, and very high scores may reflect simple label patterns. Results on this dataset are not directly comparable to the previous 614-row dataset.

### Additional smaller-loan dataset
A separate form uses Taweilo's 45,000-row synthetic Kaggle Loan Approval Classification Dataset. Annual incomes start at 8,000 and loans span 500–35,000 in undocumented currency units. These records are not merged with the asset-based source. Nine financial and credit fields are used; age, gender and education are excluded. The source lacks assets and term, so no assets are inferred and entered term affects only repayment arithmetic. Three-fold grouped CV and an 80/20 grouped holdout evaluate predeclared pipelines, followed by final refits on every source row. Identical selected-feature groups stay in one split; upstream SMOTENC generation may still make metrics optimistic. Existing histories retain source and model version. See the small-loan Model performance page for actual measured results.

### Frontend, backend and database
HTML/CSS present the responsive staff workspace. JavaScript sends validated input JSON to Flask. pandas builds the source-native frame, the three saved scikit-learn pipelines return scores, and the backend calculates their arithmetic mean. PostgreSQL (or SQLite for offline development) stores the canonical inputs, schema/model version, raw scores and staff review history. CSV batch uploads validate up to 200 rows before an atomic save. Search, filters, exports and printable reports complete the workflow.

Each application starts pending. An employee or manager explicitly records Accept, Reject or Keep pending with notes. CSRF protection and a stale-review check protect this form. Reviewer identities/roles are self-entered local-demo labels. Models do not make final lending decisions. Training-range, sparse-term and principal-only arithmetic checks are review prompts; passing them does not certify reliability.

### Conclusion and limitations
The project demonstrates three syllabus algorithms connected to a working website and database. Future work requires documented institutional data, independently established units, external validation, calibration, fairness analysis and authenticated staff access before real institutional use. Legacy records retain their original input schema and scores rather than receiving invented CIBIL or asset values.

**Source:** https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset  
**Detailed audit:** KAGGLE_DATASET.md  
**Submission:** Add student details and current screenshots; format the final printout to your faculty's requested page count.
