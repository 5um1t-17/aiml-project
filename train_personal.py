"""Train three small-loan demo models using a separate synthetic public dataset."""
import hashlib,json,joblib,numpy as np,pandas as pd,warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold,cross_val_score
from train import evaluate
from personal_loans import ROOT,FEATURES,NUMERIC,CATEGORICAL,SCHEMA,VERSION
from training_audit import checked_split,completed_run

def load_personal():
    return pd.read_csv(ROOT/'data/personal_loan_data.csv')

def training_frame(raw):
    X=raw[FEATURES].copy()
    # The publisher flags impossible ages; age is excluded. Impossible 125-year
    # employment experience becomes missing in both evaluation and final fitting.
    X.loc[~X.person_emp_exp.between(0,80),'person_emp_exp']=np.nan
    return X

def split(raw):
    X=training_frame(raw);y=raw.loan_status
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    train,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,groups))
    return X,y,groups,train,test

def train():
    warnings.filterwarnings('error',category=ConvergenceWarning)
    raw=load_personal();X,y,groups,train,test=split(raw)
    split_audit=checked_split(X,y,train,test,groups)
    if not y.isin([0,1]).all():raise ValueError('Unexpected source labels.')
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)])
    estimators={'Logistic Regression':LogisticRegression(max_iter=2500,class_weight='balanced',random_state=42),
        'Decision Tree':DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,class_weight='balanced',random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=14,min_samples_leaf=5,class_weight='balanced',random_state=42)}
    evaluation={};final={};results=[];scores=[]
    cv=StratifiedGroupKFold(3,shuffle=True,random_state=43)
    for name,estimator in estimators.items():
        print('Training small-loan',name,flush=True)
        pipeline=Pipeline([('preprocess',clone(prep)),('classifier',estimator)])
        cv_scores=cross_val_score(pipeline,X.iloc[train],y.iloc[train],groups=groups[train],cv=cv,scoring='balanced_accuracy',n_jobs=1,error_score='raise')
        fitted=clone(pipeline).fit(X.iloc[train],y.iloc[train]);probability=fitted.predict_proba(X.iloc[test])[:,1]
        evaluation[name]=fitted;scores.append(probability)
        result=dict(name=name,**evaluate(y.iloc[test],probability,fitted.predict(X.iloc[test])),cv_balanced_accuracy=float(cv_scores.mean()),cv_std=float(cv_scores.std()))
        results.append(result)
        final[name]=clone(pipeline).fit(X,y)
        print(json.dumps({key:result[key] for key in ['name','accuracy','balanced_accuracy','auc']}),flush=True)
    report=dict(model_version=VERSION,input_schema=SCHEMA,rows=len(raw),production_train_rows=len(X),train_rows=len(train),test_rows=len(test),features=FEATURES,models=results,ensemble=evaluate(y.iloc[test],np.mean(scores,axis=0)),best_model=max(results,key=lambda row:row['cv_balanced_accuracy'])['name'],source=json.loads((ROOT/'data/personal_dataset_source.json').read_text()),dataset_sha256=hashlib.sha256((ROOT/'data/personal_loan_data.csv').read_bytes()).hexdigest(),calibrated=False,synthetic=True,
        support=dict(train_rows=len(X),numeric_ranges={key:dict(min=float(X[key].min()),max=float(X[key].max())) for key in NUMERIC}),
        audit=dict(identical_feature_groups_crossing_holdout=len(set(groups[train])&set(groups[test])),original_missing=int(raw[FEATURES].isna().sum().sum()),invalid_employment_values=int((~raw.person_emp_exp.between(0,80)).sum()),excluded_source_columns=[key for key in raw.columns if key not in FEATURES+['loan_status']]),
        evaluation_note='Predeclared model parameters; three-fold grouped CV on training data. Identical selected-feature groups remain in one split. The publisher synthesized rows with SMOTENC before publication, so related/near-neighbour rows may still make internal metrics optimistic. Separate evaluation models are saved; final website models are cloned and fitted on all rows afterward. No external validation or calibrated approval probability.')
    report['training_run']=completed_run(evaluation,final,split_audit)
    joblib.dump(evaluation,ROOT/'models/personal-evaluation-pipelines.joblib');joblib.dump(final,ROOT/'models/personal-pipelines.joblib')
    (ROOT/'models/personal-metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':train()
