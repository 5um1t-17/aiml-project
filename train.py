"""Evaluate on a holdout, then refit the three selected pipelines on every source row."""
from pathlib import Path
import hashlib,json,joblib,numpy as np,pandas as pd,warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate,GridSearchCV
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,brier_score_loss,confusion_matrix,roc_curve
from features import KaggleFeatureBuilder,KAGGLE_ENGINEERED
from loan_schema import FEATURES,NUMERIC,CATEGORICAL,ASSETS,DATASET_PATH,SCHEMA_ID
from training_audit import checked_split,completed_run
ROOT=Path(__file__).parent
MODEL_VERSION='kaggle-archit-full-v2'

def load_data():
    data=pd.read_csv(DATASET_PATH)
    data.columns=data.columns.str.strip()
    for key in CATEGORICAL+['loan_status']:data[key]=data[key].str.strip()
    if not set(FEATURES+['loan_id','loan_status']).issubset(data.columns):raise ValueError('Kaggle CSV is missing required columns.')
    if not data.loan_status.isin(['Approved','Rejected']).all():raise ValueError('Unexpected target labels.')
    return data

def make_pipeline(estimator,engineered=False):
    prep=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC+(KAGGLE_ENGINEERED if engineered else [])),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)])
    return Pipeline(([('features',KaggleFeatureBuilder())] if engineered else [])+[('preprocess',prep),('classifier',estimator)])

def evaluate(y,probabilities,prediction=None):
    if prediction is None:prediction=(probabilities>=.5).astype(int)
    fpr,tpr,_=roc_curve(y,probabilities)
    return dict(accuracy=float(accuracy_score(y,prediction)),balanced_accuracy=float(balanced_accuracy_score(y,prediction)),precision=float(precision_score(y,prediction,zero_division=0)),recall=float(recall_score(y,prediction)),decline_recall=float(recall_score(y,prediction,pos_label=0)),f1=float(f1_score(y,prediction)),auc=float(roc_auc_score(y,probabilities)),brier=float(brier_score_loss(y,probabilities)),confusion=confusion_matrix(y,prediction).tolist(),roc={'fpr':fpr.tolist(),'tpr':tpr.tolist()})

def build_support(frame):
    return dict(annual_income_min=float(frame.income_annum.min()),annual_income_max=float(frame.income_annum.max()),numeric_ranges={key:dict(min=float(frame[key].min()),max=float(frame[key].max())) for key in NUMERIC},term_counts={str(int(key)):int(value) for key,value in frame.loan_term.value_counts().items()},rare_term_warning_count=10,train_rows=len(frame),term_unit='years',term_unit_status='Explicit user-selected project assumption; not specified by publisher metadata.')

def refit_all_rows(evaluation_models,X,y):
    """Clone fixed selected pipelines so evaluation models retain their unseen holdout."""
    final_models={}
    for name,pipeline in evaluation_models.items():
        print(f'Refitting {name} on all {len(X)} rows',flush=True)
        final_models[name]=clone(pipeline).fit(X,y)
    return final_models

def train():
    warnings.filterwarnings('error',category=ConvergenceWarning)
    raw=load_data();X=raw[FEATURES].copy();y=raw.loan_status.map({'Rejected':0,'Approved':1})
    negative={key:int(X[key].lt(0).sum()) for key in ASSETS}
    for key in ASSETS:X.loc[X[key]<0,key]=np.nan
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    split_audit=checked_split(X,y,X_train.index.to_numpy(),X_test.index.to_numpy())
    estimators={'Logistic Regression':LogisticRegression(max_iter=3000,random_state=42),'Decision Tree':DecisionTreeClassifier(max_depth=6,min_samples_leaf=5,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=42)}
    grids={'Logistic Regression':{'classifier__C':[.1,1,10],'classifier__class_weight':[None,'balanced']},'Decision Tree':{'classifier__max_depth':[4,8,None],'classifier__min_samples_leaf':[2,5],'classifier__class_weight':[None,'balanced']},'Random Forest':{'classifier__max_depth':[10,None],'classifier__min_samples_leaf':[2,5],'classifier__class_weight':[None,'balanced']}}
    cv=StratifiedKFold(5,shuffle=True,random_state=42);scoring={'balanced_accuracy':'balanced_accuracy','f1':'f1'}
    results=[];fitted={};test_scores=[];baseline_scores=[]
    for name,estimator in estimators.items():
        print('Training',name,flush=True)
        baseline=make_pipeline(clone(estimator));bcv=cross_validate(baseline,X_train,y_train,cv=cv,scoring=scoring)
        baseline.fit(X_train,y_train);bp=baseline.predict_proba(X_test)[:,1];baseline_scores.append(bp)
        search=GridSearchCV(make_pipeline(clone(estimator),True),grids[name],scoring=scoring,refit='balanced_accuracy',cv=cv,n_jobs=1,error_score='raise');search.fit(X_train,y_train)
        improved=search.best_score_>bcv['test_balanced_accuracy'].mean();chosen=search.best_estimator_ if improved else baseline
        probability=chosen.predict_proba(X_test)[:,1];test_scores.append(probability);fitted[name]=chosen
        row=dict(name=name,**evaluate(y_test,probability,chosen.predict(X_test)),cv_f1=float(search.cv_results_['mean_test_f1'][search.best_index_] if improved else bcv['test_f1'].mean()),cv_std=float(search.cv_results_['std_test_f1'][search.best_index_] if improved else bcv['test_f1'].std()),cv_balanced_accuracy=float(search.best_score_ if improved else bcv['test_balanced_accuracy'].mean()),baseline_cv_balanced_accuracy=float(bcv['test_balanced_accuracy'].mean()),pipeline_variant='Source + financial ratios' if improved else 'Source features retained',selected_params=search.best_params_ if improved else {})
        results.append(row);print(json.dumps({key:row[key] for key in ['name','accuracy','balanced_accuracy','decline_recall','pipeline_variant']}),flush=True)
    support=build_support(X_train)
    source=json.loads((ROOT/'data/dataset_source.json').read_text(encoding='utf-8'))
    report=dict(model_version=SCHEMA_ID,input_schema=SCHEMA_ID,rows=len(raw),train_rows=len(X_train),test_rows=len(X_test),features=FEATURES,best_model=max(results,key=lambda item:item['cv_balanced_accuracy'])['name'],approved=int(y.sum()),declined=int((y==0).sum()),missing=int(X.isna().sum().sum()),models=results,ensemble=evaluate(y_test,np.mean(test_scores,axis=0)),baseline_ensemble=evaluate(y_test,np.mean(baseline_scores,axis=0)),baseline_accuracy=float((y_test==y_train.mode()[0]).mean()),seed=42,support=support,source=source,dataset_sha256=hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),audit=dict(exact_duplicate_rows=int(raw.duplicated().sum()),duplicate_ids=int(raw.loan_id.duplicated().sum()),conflicting_feature_groups=int(raw.groupby(FEATURES,dropna=False).loan_status.nunique().gt(1).sum()),original_missing=int(raw[FEATURES].isna().sum().sum()),negative_asset_values=negative,negative_asset_treatment='Negative asset values replaced with missing; fold-fitted median imputation. All target labels and rows retained.',missing_by_feature={key:int(value) for key,value in X.isna().sum().items()}),selection_note='Training-only 5-fold CV balanced accuracy chooses source/engineered variants and hyperparameters. CV values are selection estimates, not external validation.',holdout_note='New stratified 20% holdout for this Kaggle CSV, excluded from fitting and parameter selection. No comparison of accuracy against the different 614-row mortgage dataset is valid. No external bank validation.',calibrated=False)
    final_models=refit_all_rows(fitted,X,y)
    report.update(model_version=MODEL_VERSION,production_train_rows=len(X),evaluation_model_version=SCHEMA_ID,
        evaluation_artifact='models/kaggle-evaluation-pipelines.joblib',evaluation_support=support,support=build_support(X),
        deployment_note='The website uses selected pipelines refitted on all source rows after evaluation. Reported test metrics belong to separate 3415-row evaluation models, not the all-row refits. Smaller amounts absent from the source remain outside training coverage.',
        holdout_note='The 854-row holdout is excluded from fitting and parameter selection of the evaluation models. After evaluation, fixed selected pipelines are cloned and refitted on all 4269 source rows for website inference. Final all-row models do not have a separate unseen-test accuracy measurement. No external bank validation.')
    report['training_run']=completed_run(fitted,final_models,split_audit)
    joblib.dump(fitted,ROOT/'models/kaggle-evaluation-pipelines.joblib')
    joblib.dump(final_models,ROOT/'models/kaggle-pipelines.joblib')
    (ROOT/'models/kaggle-metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Equal-weight ensemble:',json.dumps({key:report['ensemble'][key] for key in ['accuracy','balanced_accuracy','decline_recall','auc']}),flush=True)
    return report

if __name__=='__main__':train()
