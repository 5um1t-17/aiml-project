"""Canonical inputs for the user-selected Kaggle dataset."""
from pathlib import Path
SCHEMA_ID = 'kaggle-archit-v1'
NUMERIC = ['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
CATEGORICAL = ['education','self_employed']
FEATURES = NUMERIC + CATEGORICAL
ASSETS = ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
LABELS = dict(no_of_dependents='Dependents',income_annum='Annual income',loan_amount='Loan amount',loan_term='Loan term (years assumed)',cibil_score='CIBIL score',residential_assets_value='Residential assets',commercial_assets_value='Commercial assets',luxury_assets_value='Luxury assets',bank_asset_value='Bank assets',education='Education',self_employed='Self-employed',total_assets='Total assets',loan_to_income='Loan / annual income',assets_to_loan='Assets / loan',principal_to_income='Annual principal / income',log_income='Log annual income',log_loan_amount='Log loan amount')
DATASET_PATH = Path(__file__).parent/'data/kaggle_loan_approval.csv'
