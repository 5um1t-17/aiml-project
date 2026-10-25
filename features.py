"""Features computed within each training fold, without modifying target labels."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED = ['TotalIncome', 'LoanToAnnualIncome', 'PrincipalToIncome', 'LogTotalIncome', 'LogLoanAmount']

class LoanFeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.medians_ = X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']].median()
        return self

    def transform(self, X):
        output = X.copy()
        numeric = output[self.medians_.index].fillna(self.medians_)
        total = (numeric.ApplicantIncome + numeric.CoapplicantIncome).clip(lower=1)
        amount = numeric.LoanAmount * 1000
        output['TotalIncome'] = total
        output['LoanToAnnualIncome'] = amount / (total * 12)
        output['PrincipalToIncome'] = amount / numeric.Loan_Amount_Term.clip(lower=1) / total
        output['LogTotalIncome'] = np.log1p(total)
        output['LogLoanAmount'] = np.log1p(amount)
        return output

KAGGLE_ASSETS = ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
KAGGLE_ENGINEERED = ['total_assets','loan_to_income','assets_to_loan','principal_to_income','log_income','log_loan_amount']

class KaggleFeatureBuilder(TransformerMixin, BaseEstimator):
    """Ratios learned from source columns; term-years is an explicit project assumption."""
    def fit(self, X, y=None):
        self.columns_ = ['income_annum','loan_amount','loan_term'] + KAGGLE_ASSETS
        self.medians_ = X[self.columns_].median()
        return self

    def transform(self, X):
        output = X.copy()
        numeric = output[self.columns_].fillna(self.medians_)
        income = numeric.income_annum.clip(lower=1)
        amount = numeric.loan_amount.clip(lower=1)
        output['total_assets'] = numeric[KAGGLE_ASSETS].sum(axis=1)
        output['loan_to_income'] = amount / income
        output['assets_to_loan'] = output.total_assets / amount
        output['principal_to_income'] = amount / numeric.loan_term.clip(lower=1) / income
        output['log_income'] = np.log1p(income)
        output['log_loan_amount'] = np.log1p(amount)
        return output
