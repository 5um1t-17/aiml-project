"""Separate small-loan schema: never merge incompatible sources or invent asset data."""
from pathlib import Path
import math
import pandas as pd
from amounts import format_amount

ROOT=Path(__file__).parent
SCHEMA='personal-taweilo-v1'
VERSION='personal-taweilo-full-v1'
NUMERIC=['person_income','loan_amnt','loan_int_rate','person_emp_exp','cb_person_cred_hist_length','credit_score']
CATEGORICAL=['person_home_ownership','loan_intent','previous_loan_defaults_on_file']
FEATURES=NUMERIC+CATEGORICAL
LABELS=dict(person_income='Annual income',loan_amnt='Loan amount',loan_int_rate='Interest rate (%)',person_emp_exp='Employment experience (years)',cb_person_cred_hist_length='Credit history (years)',credit_score='Credit score',person_home_ownership='Home ownership',loan_intent='Loan purpose',previous_loan_defaults_on_file='Previous defaults',loan_term='Term for arithmetic only (years)')
MONEY=['person_income','loan_amnt']
CHOICES=dict(person_home_ownership=['RENT','OWN','MORTGAGE','OTHER'],loan_intent=['PERSONAL','EDUCATION','MEDICAL','VENTURE','DEBTCONSOLIDATION','HOMEIMPROVEMENT'],previous_loan_defaults_on_file=['No','Yes'])
LIMITS=dict(person_income=(1,10000000000),loan_amnt=(1,100000000000),loan_int_rate=(0,100),person_emp_exp=(0,80),cb_person_cred_hist_length=(0,80),credit_score=(300,900),loan_term=(1,40))
EXAMPLE=dict(person_income=20000,loan_amnt=10000,loan_int_rate=12,person_emp_exp=2,cb_person_cred_hist_length=3,credit_score=650,person_home_ownership='RENT',loan_intent='PERSONAL',previous_loan_defaults_on_file='No',loan_term=4)

def validate_personal(data):
    if not isinstance(data,dict):raise ValueError('Submit the small-loan fields.')
    clean={}
    for key,(low,high) in LIMITS.items():
        try:
            if isinstance(data[key],bool):raise ValueError()
            value=float(data[key])
        except (ValueError,KeyError,TypeError):raise ValueError(f'{LABELS[key]} must be a number.') from None
        if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{LABELS[key]} must be between {low} and {high}.')
        if key not in MONEY+['loan_int_rate'] and not value.is_integer():raise ValueError(f'{LABELS[key]} must be a whole number.')
        clean[key]=value
    for key,choices in CHOICES.items():
        value=str(data.get(key,'')).strip()
        if value not in choices:raise ValueError(f'Choose a valid {LABELS[key].lower()}.')
        clean[key]=value
    return clean

def run_personal(models,rows):
    frame=pd.DataFrame(rows,columns=FEATURES);outputs=[[] for _ in rows]
    for name,model in models.items():
        for index,(probability,decision) in enumerate(zip(model.predict_proba(frame)[:,1],model.predict(frame))):
            outputs[index].append(dict(name=name,approved=bool(decision),probability=float(probability)))
    return outputs

def personal_support(data,report):
    income=data['person_income']/12;principal=data['loan_amnt']/(data['loan_term']*12);flags=[]
    for key,bounds in report['support']['numeric_ranges'].items():
        if not bounds['min']<=data[key]<=bounds['max']:
            flags.append(dict(code='feature_outside_training',message=f'{LABELS[key]} is outside this dataset\'s training range ({bounds["min"]:,.2f}–{bounds["max"]:,.2f}).'))
    if principal>income:flags.append(dict(code='principal_exceeds_income',message='Principal repayment alone exceeds the monthly income equivalent. Review repayment capacity and the entered values.'))
    return dict(requires_review=bool(flags),flags=flags,monthly_income=income,income_label='Monthly income equivalent',minimum_monthly_principal=principal,principal_income_ratio=principal/income,term_examples=None,
        note='This synthetic dataset has no loan term. Your term in years is used only for this principal-only arithmetic and is excluded from model prediction. Interest, debts and living expenses are excluded. Source units have no documented currency; no rupee/dollar conversion is implied.')

def profile(data):
    entries=[]
    for key in FEATURES+['loan_term']:
        value=data[key]
        if key in MONEY:value=format_amount(value)
        elif key in NUMERIC+['loan_term']:value=f'{value:g}'
        entries.append((LABELS[key],value))
    return entries,{LABELS[key]:data[key] for key in MONEY}
