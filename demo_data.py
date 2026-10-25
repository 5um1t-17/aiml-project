"""Four fictional display fixtures, separate from the labelled training CSV."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from database import connect, insert_application,begin_write
from postgres_backend import is_postgres
from prediction import validate, run_models

DEMOS = [
    ('Standard Kaggle-format example',dict(no_of_dependents=2,income_annum=1200000,loan_amount=3000000,loan_term=8,cibil_score=740,residential_assets_value=2000000,commercial_assets_value=0,luxury_assets_value=4000000,bank_asset_value=800000,education='Graduate',self_employed='No')),
    ('Lower credit-score example',dict(no_of_dependents=1,income_annum=1500000,loan_amount=4000000,loan_term=10,cibil_score=420,residential_assets_value=2000000,commercial_assets_value=1000000,luxury_assets_value=4000000,bank_asset_value=500000,education='Graduate',self_employed='Yes')),
    ('Outside training support',dict(no_of_dependents=0,income_annum=6000,loan_amount=20000,loan_term=1,cibil_score=650,residential_assets_value=0,commercial_assets_value=0,luxury_assets_value=0,bank_asset_value=0,education='Not Graduate',self_employed='No')),
    ('Asset-backed example',dict(no_of_dependents=1,income_annum=3000000,loan_amount=8000000,loan_term=12,cibil_score=800,residential_assets_value=5000000,commercial_assets_value=2000000,luxury_assets_value=8000000,bank_asset_value=2000000,education='Graduate',self_employed='Yes')),
]

# Balanced display fixtures for the complete two-source website. Amounts are
# realistic relative to each source; neither source documents a currency.
# Staff reviews below are explicitly fictional workflow examples, never model-
# generated decisions about real applicants.
PROJECT_DEMOS = [
    dict(source='personal',label='Regular income / personal loan',data=dict(person_income=48000,loan_amnt=12000,loan_int_rate=12,person_emp_exp=5,cb_person_cred_hist_length=7,credit_score=710,person_home_ownership='RENT',loan_intent='PERSONAL',previous_loan_defaults_on_file='No',loan_term=3),
        review=dict(decision='accepted',role='Credit Manager',notes='SIMULATED DEMO REVIEW: A fictional credit manager recorded acceptance after reviewing the sample income, credit history and repayment details. This is an example staff workflow, not a real lending decision.')),
    dict(source='personal',label='Previous defaults / credit review',data=dict(person_income=54000,loan_amnt=20000,loan_int_rate=17,person_emp_exp=3,cb_person_cred_hist_length=4,credit_score=510,person_home_ownership='RENT',loan_intent='DEBTCONSOLIDATION',previous_loan_defaults_on_file='Yes',loan_term=3),
        review=dict(decision='rejected',role='Credit Committee',notes='SIMULATED DEMO REVIEW: A fictional credit committee recorded rejection after reviewing the sample previous defaults and credit history. This demonstrates a recorded staff decision, not a real lending decision.')),
    dict(source='assets',label='Asset-backed loan / pending staff review',data=dict(no_of_dependents=2,income_annum=900000,loan_amount=1800000,loan_term=6,cibil_score=780,residential_assets_value=2400000,commercial_assets_value=0,luxury_assets_value=2700000,bank_asset_value=1200000,education='Graduate',self_employed='No')),
    dict(source='assets',label='Larger loan / repayment-capacity review',data=dict(no_of_dependents=2,income_annum=600000,loan_amount=4800000,loan_term=4,cibil_score=620,residential_assets_value=1800000,commercial_assets_value=200000,luxury_assets_value=1200000,bank_asset_value=400000,education='Graduate',self_employed='Yes')),
]


def insert_project_demos(connection,models,metrics,personal_models,personal_metrics):
    from personal_loans import validate_personal,run_personal
    ids=[]
    for example in PROJECT_DEMOS:
        if example['source']=='personal':
            clean=validate_personal(example['data']);predictions=run_personal(personal_models,[clean])[0];report=personal_metrics
        else:
            clean=validate(example['data']);predictions=run_models(models,[clean])[0];report=metrics
        key=insert_application(connection,clean,predictions,report['best_model'],report['model_version'],example['label'])
        ids.append(key)
        if 'review' in example:
            review=example['review']
            connection.execute('''INSERT INTO staff_reviews
                (application_id,reviewer_name,reviewer_role,decision,notes,created_at)
                VALUES (?,?,?,?,?,?)''',(key,'Demo '+review['role'],review['role'],review['decision'],review['notes'],datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')))
    return ids

def insert_demos(connection, models, metrics, personal_models=None, personal_metrics=None):
    if personal_models is not None and personal_metrics is not None:
        return insert_project_demos(connection,models,metrics,personal_models,personal_metrics)
    if personal_models is not None or personal_metrics is not None:
        raise ValueError('Provide both the personal-loan models and metrics.')
    rows = [validate(data) for _, data in DEMOS]
    outputs = run_models(models, rows)
    return [insert_application(connection, data, predictions, metrics['best_model'],
                metrics.get('model_version','legacy'), label)
            for (label,_),data,predictions in zip(DEMOS,rows,outputs)]

def seed_demo_if_empty(path, models, metrics, personal_models=None, personal_metrics=None):
    """Initialize an empty workspace once; never reset populated history."""
    with closing(connect(path)) as connection, connection:
        begin_write(connection)
        if connection.execute('SELECT COUNT(*) FROM applications').fetchone()[0]:
            return []
        return insert_demos(connection, models, metrics, personal_models, personal_metrics)

def reset_with_backup(path, backup_directory, models, metrics, personal_models=None, personal_metrics=None):
    """Explicit maintenance operation; recoverable DB snapshot before reset."""
    if is_postgres(path):raise ValueError('This reset helper is SQLite-only; back up PostgreSQL with its provider before maintenance.')
    directory = Path(backup_directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = 'loanlens-before-demo-reset-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.db'
    backup = directory / filename
    with closing(connect(path)) as source, closing(sqlite3.connect(backup)) as destination:
        source.backup(destination)
        assert destination.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    with closing(connect(path)) as connection, connection:
        begin_write(connection)
        connection.execute('DELETE FROM staff_reviews')
        connection.execute('DELETE FROM predictions')
        connection.execute('DELETE FROM applications')
        connection.execute("DELETE FROM sqlite_sequence WHERE name IN ('applications','predictions','staff_reviews')")
        ids = insert_demos(connection, models, metrics, personal_models, personal_metrics)
    return backup, ids

def refresh_unreviewed_demos(path, models, metrics):
    """Replace only obsolete, unreviewed demo fixtures; retain all user history."""
    from loan_schema import SCHEMA_ID
    from personal_loans import SCHEMA as PERSONAL_SCHEMA
    with closing(connect(path)) as connection, connection:
        begin_write(connection)
        ids = [row['id'] for row in connection.execute('''SELECT a.id FROM applications a
            WHERE a.is_demo=1 AND a.input_schema NOT IN (?,?) AND NOT EXISTS
            (SELECT 1 FROM staff_reviews r WHERE r.application_id=a.id)''',(SCHEMA_ID,PERSONAL_SCHEMA))]
        for key in ids:
            connection.execute('DELETE FROM predictions WHERE application_id=?',(key,))
            connection.execute('DELETE FROM applications WHERE id=?',(key,))
        current = connection.execute('SELECT COUNT(*) FROM applications WHERE is_demo=1 AND input_schema IN (?,?)',(SCHEMA_ID,PERSONAL_SCHEMA)).fetchone()[0]
        created = insert_demos(connection,models,metrics) if not current else []
        return dict(replaced_demo_ids=ids,new_demo_ids=created)
