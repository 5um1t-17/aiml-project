# Viva preparation

1. **Source?** Kaggle Loan-Approval-Prediction-Dataset by architsharma01, 4,269 labelled rows.
2. **Algorithms?** Logistic Regression, Decision Tree and Random Forest.
3. **Target?** Approved versus Rejected. Approved is encoded 1.
4. **Inputs?** Annual income, loan amount, term, CIBIL score, dependents, education, self-employment and four asset values.
5. **Why exclude loan_id?** It identifies rows and is not a financial predictor.
6. **Why exclude loan_status?** It is the target; including it would leak the answer.
7. **Missing/invalid data?** Negative residential-asset values become missing. Fold-fitted median imputation handles them.
8. **Why a pipeline?** Imputation, scaling, encoding and feature learning are fitted only on each training fold.
9. **What is cross-validation?** Train/validate on rotating parts of the training data; use balanced accuracy for model selection.
10. **What is balanced accuracy?** The average recall across Approved and Rejected classes.
11. **What is precision?** True approvals divided by predicted approvals.
12. **What is recall?** True approvals divided by actual approvals; rejected-label recall uses Rejected as positive instead.
13. **What is F1?** The harmonic mean of precision and recall.
14. **What is ROC AUC?** Ranking discrimination across classification thresholds.
15. **What is the average score?** The arithmetic mean of the three approval scores, with equal weights. It is not majority voting or calibrated probability.
16. **Annual or monthly income?** Annual input; divide by 12 only for displayed monthly-equivalent arithmetic.
17. **Term unit?** Years is an explicit project assumption, not a documented source fact.
18. **Why can 100% holdout accuracy be misleading?** It can reflect simple patterns in a file; it does not prove generalization to actual bank data.
19. **What does SQLite store?** Source-native JSON inputs, schema/model version, model scores and an append-only staff-review history.
20. **Who makes the final decision?** A human reviewer explicitly records acceptance/rejection; the models supply support scores only.
21. **What about previous records?** Original mortgage-format inputs and scores are preserved; missing CIBIL/assets are not fabricated.
22. **What does an explanation mean?** Exact current Logistic Regression coefficient contributions in log-odds units, not causal effects and not a decomposition of the mean ensemble.

23. **Why two datasets?** The original source lacks smaller loans/incomes. The separate Taweilo dataset covers smaller amounts but has different fields and undocumented currency, so they are never merged.
24. **Is the new source real bank data?** The publisher identifies it as synthetic and expanded with SMOTENC before publication. Its supplied target labels are retained without inventing labels.
25. **How are final models trained?** Separate evaluation models use a held-out split; after evaluation, clones with fixed choices are refitted on all source rows for website inference. Reported test accuracy belongs to the evaluation models.
26. **Does term affect the new models?** No. That source has no term; the entered years are used only for principal arithmetic.
