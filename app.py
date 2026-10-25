from pathlib import Path
import csv
import io
import json
import os
import secrets
import hmac
import joblib
from flask import Flask, render_template, request, jsonify, Response, abort, redirect, url_for, session, send_file
from itsdangerous import URLSafeSerializer, BadSignature
from loan_schema import FEATURES,SCHEMA_ID,LABELS,NUMERIC
from amounts import amount_in_words, format_amount
import personal_loans as personal
from configuration import database_target,signing_keys
from postgres_backend import is_postgres
from prediction import validate, run_models, explain_linear, summarize_predictions, assess_support, legacy_screening
from database import init_database, save_application, save_batch, query_applications, get_application, dashboard_stats, export_records, record_review,DATABASE_ERRORS,REVIEWER_ROLES

ROOT = Path(__file__).parent
app = Flask(__name__)
app.add_template_filter(amount_in_words, 'amount_words')
app.add_template_filter(format_amount, 'amount_number')
secret_key,receipt_key=signing_keys()
app.config.update(MAX_CONTENT_LENGTH=1024 * 1024,DATABASE=database_target())
app.config.update(SECRET_KEY=secret_key,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Lax',SESSION_COOKIE_SECURE=bool(os.environ.get('VERCEL')))
receipts = URLSafeSerializer(receipt_key, salt='local-batch-receipt')
# PostgreSQL tables and historical records are prepared by the explicit
# migration command, never seeded or reset during a serverless cold start.
if not is_postgres(app.config['DATABASE']):init_database(app.config['DATABASE'])
required_artifacts=('kaggle-pipelines.joblib','kaggle-metrics.json','personal-pipelines.joblib','personal-metrics.json')
missing=[name for name in required_artifacts if not (ROOT/'models'/name).is_file()]
if missing:
    raise RuntimeError('Missing trained model artifacts: '+', '.join(missing)+'. Restore the supplied runtime model files before starting the website.')
models = joblib.load(ROOT / 'models/kaggle-pipelines.joblib')
metrics = json.loads((ROOT / 'models/kaggle-metrics.json').read_text())
personal_models=joblib.load(ROOT/'models/personal-pipelines.joblib')
personal_metrics=json.loads((ROOT/'models/personal-metrics.json').read_text())
from demo_data import seed_demo_if_empty
if not is_postgres(app.config['DATABASE']):
    seed_demo_if_empty(app.config['DATABASE'], models, metrics, personal_models, personal_metrics)

def record_input(item):
    if item.get('input_schema') in (SCHEMA_ID,personal.SCHEMA):
        return json.loads(item['input_json'])
    mapping = dict(zip(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Dependents','Education','Self_Employed','Property_Area'], ['applicant_income','coapplicant_income','loan_amount','loan_term','credit_history','dependents','education','self_employed','property_area']))
    return {key:item[column] for key,column in mapping.items()}

def screen_record(item):
    data = record_input(item)
    if item.get('input_schema') == personal.SCHEMA:
        return personal.personal_support(data,personal_metrics)
    support=metrics.get('support')
    if item.get('model_version') == metrics.get('evaluation_model_version'):
        support=metrics.get('evaluation_support',support)
    return assess_support(data,support) if item.get('input_schema') == SCHEMA_ID else legacy_screening(data)

def decorate(records):
    for item in records:item['screening'] = screen_record(item)
    return records

def detail_context(item):
    clean = record_input(item);is_personal=item.get('input_schema') == personal.SCHEMA
    current = item.get('input_schema') in (SCHEMA_ID,personal.SCHEMA)
    screening = screen_record(item)
    profile_amounts = {}
    if is_personal:
        profile,profile_amounts=personal.profile(clean)
    elif current:
        profile = []
        for key in FEATURES:
            value = clean[key]
            if key == 'loan_term':value = f'{int(value)} years (project assumption)'
            elif key in ('no_of_dependents','cibil_score'):value = str(int(value))
            elif key in NUMERIC:
                profile_amounts[LABELS[key]] = value
                value = format_amount(value)
            profile.append((LABELS[key],value))
    else:
        profile = [('Monthly applicant income',item['applicant_income']),('Monthly co-applicant income',item['coapplicant_income']),('Full loan amount',f'{item["loan_amount"]*1000:,.2f}'),('Term / months',item['loan_term']),('Credit-history criteria',item['credit_history']),('Dependents',item['dependents']),('Education',item['education']),('Self-employed',item['self_employed']),('Property area',item['property_area'])]
        profile_amounts = {'Monthly applicant income':item['applicant_income'], 'Monthly co-applicant income':item['coapplicant_income'], 'Full loan amount':item['loan_amount']*1000}
        profile = [(label,format_amount(profile_amounts[label]) if label in profile_amounts else value) for label,value in profile]
    explanation=[]
    if is_personal:explanation=explain_linear(personal_models['Logistic Regression'],clean,personal.FEATURES,personal.CATEGORICAL,personal.LABELS)
    elif current:explanation=explain_linear(models['Logistic Regression'],clean)
    return dict(page='history',item=item,screening=screening,summary=summarize_predictions(item['predictions'],screening),
        explanation=explanation,profile=profile,profile_amounts=profile_amounts,current_schema=current,is_personal=is_personal)

@app.context_processor
def common():
    return dict(metrics=metrics,personal_metrics=personal_metrics,csrf_token=session.setdefault('csrf_token', secrets.token_hex(32)))

@app.before_request
def local_mutations():
    if request.method == 'POST':
        origin = request.headers.get('Origin')
        if (origin and origin != request.host_url.rstrip('/')) or request.headers.get('Sec-Fetch-Site') == 'cross-site':
            return jsonify(error='Submit applications from this website.'), 403

@app.errorhandler(413)
def oversized(error):
    if request.path.startswith('/api/'):
        return jsonify(error='The request is too large. Maximum upload size is 1 MB.'), 413
    template='personal_batch.html' if request.path.startswith('/batch/personal') else 'batch.html'
    return render_template(template, page='batch', error='File is too large. Maximum upload size is 1 MB.'), 413

@app.get('/')
def home():
    return render_template('dashboard.html', page='dashboard', stats=dashboard_stats(app.config['DATABASE']),
        recent=decorate(query_applications(app.config['DATABASE'], per_page=6)['records']))

@app.get('/health')
def health():
    return jsonify(service='loanlens',status='ready',database='postgresql' if is_postgres(app.config['DATABASE']) else 'sqlite')

@app.get('/predict')
def prediction():
    if request.args.get('dataset','personal') == 'personal':
        return render_template('personal.html',page='predict',example=personal.EXAMPLE,choices=personal.CHOICES)
    return render_template('index.html', page='predict')

@app.get('/comparison')
def comparison():
    if request.args.get('dataset') == 'personal':
        return render_template('personal_comparison.html',page='compare')
    return render_template('comparison.html', page='compare')

@app.get('/about')
def about():
    return render_template('about.html', page='about')

@app.get('/history')
def history():
    search = request.args.get('q', '').strip()[:100]
    status = request.args.get('status', '')
    try:
        saved = query_applications(app.config['DATABASE'], page=request.args.get('page', 1, type=int), search=search, status=status)
        decorate(saved['records'])
    except DATABASE_ERRORS:
        app.logger.error('Could not read application history')
        return render_template('history.html', page='history', saved=None), 503
    return render_template('history.html', page='history', saved=saved, search=search, status=status)

@app.post('/api/predict')
def predict():
    try:
        clean = validate(request.get_json(silent=True))
        predictions = run_models(models, [clean])[0]
        screening = assess_support(clean, metrics.get('support'))
        application_id = save_application(app.config['DATABASE'], clean, predictions, metrics['best_model'], metrics.get('model_version','legacy'))
        return jsonify(predictions=predictions, selected=metrics['best_model'], application_id=application_id,
                       summary=summarize_predictions(predictions, screening), screening=screening)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except DATABASE_ERRORS:
        app.logger.error('Could not save application and predictions')
        return jsonify(error='Your application could not be saved. Please try again.'), 503

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'GET':
        receipt = request.args.get('receipt')
        if receipt:
            try:
                ids = receipts.loads(receipt)
                return render_template('batch.html', page='batch', imported=len(ids), first_id=ids[0], last_id=ids[-1])
            except (BadSignature, IndexError, TypeError):
                return render_template('batch.html', page='batch', error='This upload receipt has expired. Your saved records are still on the Applications page.'), 400
        return render_template('batch.html', page='batch')
    file = request.files.get('file')
    try:
        if not file or not file.filename.lower().endswith('.csv'):
            raise ValueError('Choose a CSV file to upload.')
        reader = csv.DictReader(io.StringIO(file.read().decode('utf-8-sig'), newline=''))
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if not reader.fieldnames or not set(FEATURES).issubset(reader.fieldnames):
            raise ValueError('Missing required columns. Download the sample CSV for the exact headers.')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('CSV headers must be unique.')
        rows = []
        for line, row in enumerate(reader, start=2):
            if len(rows) == 200:
                raise ValueError('Maximum 200 applications per upload.')
            try:
                rows.append(validate(row))
            except ValueError as error:
                raise ValueError(f'CSV line {line}: {error}') from None
        if not rows:
            raise ValueError('CSV contains no application rows.')
        outputs = run_models(models, rows)
        ids = save_batch(app.config['DATABASE'], list(zip(rows, outputs)), metrics['best_model'], metrics.get('model_version','legacy'))
        return redirect(url_for('batch', receipt=receipts.dumps(ids)), code=303)
    except (UnicodeError, csv.Error):
        return render_template('batch.html', page='batch', error='Use a valid UTF-8 CSV file.'), 400
    except ValueError as error:
        return render_template('batch.html', page='batch', error=str(error)), 400
    except DATABASE_ERRORS:
        app.logger.error('CSV could not be saved')
        return render_template('batch.html', page='batch', error='Upload could not be saved. No partial batch was kept.'), 503

def csv_download(rows, headers, filename):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction='ignore')
    writer.writeheader()
    writer.writerows({key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')) else value)
                     for key, value in row.items()} for row in rows)
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename="{filename}"'})

@app.get('/sample.csv')
def sample_csv():
    from demo_data import DEMOS
    rows = [data for _,data in DEMOS]
    return csv_download(rows, FEATURES, 'loanlens-demo-applications.csv')

@app.get('/dataset.csv')
def training_dataset():
    return send_file(ROOT/'data/kaggle_loan_approval.csv',as_attachment=True,download_name='loan_approval_dataset.csv')

@app.get('/export.csv')
def export_csv():
    records = list(export_records(app.config['DATABASE'], search=request.args.get('q', '').strip()[:100], status=request.args.get('status', '')))
    for row in records:
        screening = screen_record(row)
        row['model_review_required'] = screening['requires_review']
        row['model_review_flags'] = ' | '.join(flag['message'] for flag in screening['flags'])
        payload = record_input(row)
        if row.get('input_schema') in (SCHEMA_ID,personal.SCHEMA):
            if row.get('input_schema') == personal.SCHEMA:
                for key in FEATURES:
                    if key not in payload:row[key]=''
            row.update(payload)
        else:
            row['legacy_input_json'] = json.dumps(payload,sort_keys=True)
            row['legacy_monthly_applicant_income'] = row['applicant_income']
            row['legacy_monthly_coapplicant_income'] = row['coapplicant_income']
            row['legacy_loan_amount'] = row['loan_amount']*1000
            row['legacy_term_months'] = row['loan_term']
            for key in FEATURES:row[key] = ''
    headers = ['id','created_at','input_schema','is_demo','demo_label','model_version'] + FEATURES + [key for key in personal.FEATURES if key not in FEATURES] + ['legacy_input_json','legacy_monthly_applicant_income','legacy_monthly_coapplicant_income','legacy_loan_amount','legacy_term_months','average_probability','model_review_required','model_review_flags','review_status','reviewer_name','reviewer_role','review_notes','reviewed_at','model_name','approved','probability']
    return csv_download(records, headers, 'loanlens-predictions.csv')

@app.get('/applications/<int:application_id>')
def detail(application_id):
    item = get_application(app.config['DATABASE'], application_id)
    if item is None:
        abort(404)
    return render_template('detail.html', **detail_context(item))

@app.post('/applications/<int:application_id>/review')
def review(application_id):
    token = request.form.get('csrf_token', '')
    if not token or not hmac.compare_digest(token, session.get('csrf_token', '')):
        abort(400, description='This review form expired. Reload the application and try again.')
    item = get_application(app.config['DATABASE'], application_id)
    if item is None:
        abort(404)
    try:
        reviewer = request.form.get('reviewer_name', '').strip()
        role = request.form.get('reviewer_role', '')
        decision = request.form.get('decision', '')
        notes = request.form.get('notes', '').strip()
        if not 2 <= len(reviewer) <= 80:
            raise ValueError('Enter the reviewer name (2–80 characters).')
        if role not in REVIEWER_ROLES or decision not in ('accepted', 'rejected', 'pending'):
            raise ValueError('Choose a reviewer role and a staff decision.')
        if not 10 <= len(notes) <= 2000:
            raise ValueError('Add review notes (10–2,000 characters) explaining your decision.')
        if request.form.get('confirmed') != 'yes':
            raise ValueError('Confirm that you reviewed the application independently of the model scores.')
        try:
            previous = int(request.form.get('expected_review_id', ''))
        except ValueError:
            raise ValueError('Reload the application before recording a decision.') from None
        record_review(app.config['DATABASE'], application_id, reviewer, role, decision, notes, previous)
        return redirect(url_for('detail', application_id=application_id) + '#staff-review', code=303)
    except ValueError as error:
        item = get_application(app.config['DATABASE'], application_id)
        return render_template('detail.html', **detail_context(item), review_error=str(error)), 400
    except DATABASE_ERRORS:
        app.logger.error('Could not save staff review')
        return render_template('detail.html', **detail_context(item), review_error='The staff decision could not be saved. Please try again.'), 503

from personal_routes import register_personal_routes
register_personal_routes(app,personal_models,personal_metrics,receipts,csv_download)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
