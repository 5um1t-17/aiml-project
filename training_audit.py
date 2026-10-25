"""Fail-fast split checks and reproducible provenance for local model rebuilds."""
from datetime import datetime,timezone
import platform
import joblib,numpy as np,pandas as pd,sklearn


def checked_split(X,y,train,test,groups=None):
    train=np.asarray(train);test=np.asarray(test)
    if len(set(train))!=len(train) or len(set(test))!=len(test):
        raise ValueError('Repeated rows within a training or test split.')
    if set(train)&set(test) or set(train)|set(test)!=set(range(len(X))):
        raise ValueError('Training/test rows overlap or do not cover the source.')
    if y.isna().any() or set(y.unique())!={0,1}:
        raise ValueError('Expected both binary source-label classes, with no missing labels.')
    if {'loan_id','loan_status'}.intersection(X.columns):
        raise ValueError('ID or target leaked into model features.')
    if groups is None:groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    overlap=len(set(groups[train])&set(groups[test]))
    if overlap:raise ValueError('Identical model inputs cross the holdout split.')
    counts=lambda rows:{str(int(key)):int(count) for key,count in y.iloc[rows].value_counts().sort_index().items()}
    if len(counts(train))!=2 or len(counts(test))!=2:raise ValueError('Each split must contain both source classes.')
    return dict(seed=42,source_rows=len(X),train_rows=len(train),test_rows=len(test),
        overlapping_row_indices=0,identical_feature_groups_crossing_holdout=overlap,
        train_label_counts=counts(train),test_label_counts=counts(test),target_and_id_excluded=True)


def completed_run(evaluation_models,final_models,split_audit):
    fitted={}
    for name in final_models:
        count=lambda model:int(model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].n_samples_seen_)
        eval_rows=count(evaluation_models[name]);final_rows=count(final_models[name])
        if eval_rows!=split_audit['train_rows'] or final_rows!=split_audit['source_rows']:
            raise ValueError(f'{name}: fitted row count does not match the intended split.')
        fitted[name]=dict(evaluation_fit_rows=eval_rows,final_fit_rows=final_rows)
    return dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),split=split_audit,
        fitted_models=fitted,convergence_warnings_are_errors=True,
        environment=dict(python=platform.python_version(),scikit_learn=sklearn.__version__,pandas=pd.__version__,numpy=np.__version__,joblib=joblib.__version__))
