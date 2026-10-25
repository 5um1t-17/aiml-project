"""Source-native validation, model inference and non-decision review diagnostics."""
import math
import pandas as pd
from loan_schema import FEATURES,NUMERIC,CATEGORICAL,ASSETS,LABELS,SCHEMA_ID
LIMITS={'no_of_dependents':(0,20),'income_annum':(1,10000000000),'loan_amount':(1,100000000000),'loan_term':(1,40),'cibil_score':(300,900),**{key:(0,100000000000) for key in ASSETS}}
CHOICES={'education':['Graduate','Not Graduate'],'self_employed':['Yes','No']}

def validate(data):
    """Amounts remain full source units; annual income and term-years are explicit."""
    if not isinstance(data,dict):raise ValueError('Submit the Kaggle application fields.')
    clean={}
    for key,(low,high) in LIMITS.items():
        try:
            if isinstance(data[key],bool):raise ValueError()
            value=float(data[key])
        except (KeyError,TypeError,ValueError):raise ValueError(f'{LABELS[key]} must be a number.') from None
        if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{LABELS[key]} must be between {low} and {high}.')
        if key in ('no_of_dependents','loan_term','cibil_score') and not value.is_integer():raise ValueError(f'{LABELS[key]} must be a whole number.')
        clean[key]=value
    for key,choices in CHOICES.items():
        value=str(data.get(key,'')).strip()
        if value not in choices:raise ValueError(f'Choose a valid {LABELS[key].lower()}: {", ".join(choices)}.')
        clean[key]=value
    return clean

def run_models(models,rows):
    frame=pd.DataFrame(rows,columns=FEATURES);outputs=[[] for _ in rows]
    for name,model in models.items():
        for index,(probability,decision) in enumerate(zip(model.predict_proba(frame)[:,1],model.predict(frame))):
            outputs[index].append(dict(name=name,approved=bool(decision),probability=float(probability)))
    return outputs

def assess_support(data,support):
    income=data['income_annum']/12;principal=data['loan_amount']/(data['loan_term']*12);flags=[]
    if principal>income:flags.append(dict(code='principal_exceeds_income',message=f'Using the stated years assumption, principal repayment alone is {principal:,.2f} per month against monthly income equivalent {income:,.2f}. Check repayment capacity and the entered values.'))
    if support:
        if not support['annual_income_min']<=data['income_annum']<=support['annual_income_max']:
            flags.append(dict(code='income_outside_training',message=f'Annual income {data["income_annum"]:,.2f} is outside the training range {support["annual_income_min"]:,.2f}–{support["annual_income_max"]:,.2f}. The models are extrapolating.'))
        count=support['term_counts'].get(str(int(data['loan_term'])),0)
        if count<support.get('rare_term_warning_count',10):flags.append(dict(code='sparse_term',message=f'This term value ({int(data["loan_term"])}, interpreted as years) occurs in only {count} of {support["train_rows"]} training records.'))
        for key,bounds in support.get('numeric_ranges',{}).items():
            if key!='income_annum' and not bounds['min']<=data[key]<=bounds['max']:flags.append(dict(code='feature_outside_training',message=f'{LABELS[key]} is outside the range observed during training.'))
    else:
        count=None;flags.append(dict(code='support_unavailable',message='Training support metadata is unavailable.'))
    return dict(requires_review=bool(flags),flags=flags,monthly_income=income,income_label='Monthly income equivalent',minimum_monthly_principal=principal,principal_income_ratio=principal/income,term_examples=count,
        note='Annual income is divided by 12. Loan term is interpreted as years by explicit project assumption because publisher metadata does not specify its unit. Principal-only repayment excludes interest, debts, living costs and other resources. These checks are review prompts, not lending decisions.')

def legacy_screening(data):
    income=float(data.get('ApplicantIncome') or 0)+float(data.get('CoapplicantIncome') or 0)
    amount=float(data.get('LoanAmount') or 0)*1000;term=float(data.get('Loan_Amount_Term') or 1)
    principal=amount/term
    return dict(requires_review=True,flags=[dict(code='legacy_schema',message='This record uses the previous mortgage input format. Its original scores are preserved; it cannot be evaluated by the new Kaggle models without the required annual income, CIBIL and asset inputs.')],monthly_income=income,income_label='Legacy combined monthly income',minimum_monthly_principal=principal,principal_income_ratio=principal/income if income else 0,term_examples=None,note='Legacy record: original monthly-income and term-in-months convention retained. No new source fields or scores have been inferred.')

def summarize_predictions(predictions,screening=None):
    if not predictions:raise ValueError('No model scores are available.')
    approval=math.fsum(item['probability'] for item in predictions)/len(predictions)
    label='Approval indication' if approval>=.5 else 'Decline indication'
    if screening and screening['requires_review']:label='Manual review required'
    return dict(approval_score=approval,decline_score=1-approval,assessment=label,model_count=len(predictions),review_status='pending')

def explain_linear(model,data,features=FEATURES,categorical=CATEGORICAL,labels=LABELS):
    frame=pd.DataFrame([data],columns=features)
    if 'features' in model.named_steps:frame=model.named_steps['features'].transform(frame)
    transformed=model.named_steps['preprocess'].transform(frame)
    if hasattr(transformed,'toarray'):transformed=transformed.toarray()
    contributions=transformed[0]*model.named_steps['classifier'].coef_[0]
    columns=model.named_steps['preprocess'].transformers_[0][2]
    groups=[(key,float(contributions[i])) for i,key in enumerate(columns)]
    encoder=model.named_steps['preprocess'].named_transformers_['category'].named_steps['encode'];offset=len(columns)
    for key,categories in zip(categorical,encoder.categories_):
        groups.append((key,float(contributions[offset:offset+len(categories)].sum())));offset+=len(categories)
    maximum=max(abs(value) for _,value in groups) or 1
    return [dict(label=labels[key],contribution=value,width=abs(value)/maximum*100) for key,value in sorted(groups,key=lambda item:abs(item[1]),reverse=True)]
