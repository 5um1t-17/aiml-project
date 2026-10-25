"""Small-loan endpoints use their own models, schema and source units."""
import csv,io
from flask import request,render_template,jsonify,redirect,url_for,send_file
from itsdangerous import BadSignature
import personal_loans as source
from database import save_application,save_batch,DATABASE_ERRORS
from prediction import summarize_predictions

def register_personal_routes(app,models,report,receipts,csv_download):
    @app.post('/api/predict/personal')
    def predict_personal():
        try:
            clean=source.validate_personal(request.get_json(silent=True))
            predictions=source.run_personal(models,[clean])[0]
            screening=source.personal_support(clean,report)
            key=save_application(app.config['DATABASE'],clean,predictions,report['best_model'],source.VERSION)
            return jsonify(predictions=predictions,application_id=key,summary=summarize_predictions(predictions,screening),screening=screening,model_version=source.VERSION)
        except ValueError as error:return jsonify(error=str(error)),400
        except DATABASE_ERRORS:
            app.logger.error('Small-loan application could not be saved')
            return jsonify(error='Your application could not be saved. Please try again.'),503

    @app.route('/batch/personal',methods=['GET','POST'])
    def batch_personal():
        if request.method=='GET':
            receipt=request.args.get('receipt')
            try:ids=receipts.loads(receipt) if receipt else []
            except BadSignature:return render_template('personal_batch.html',page='batch',error='Receipt expired; saved applications remain in history.'),400
            return render_template('personal_batch.html',page='batch',imported=len(ids))
        try:
            file=request.files.get('file')
            if not file or not file.filename.lower().endswith('.csv'):raise ValueError('Choose a CSV file.')
            reader=csv.DictReader(io.StringIO(file.read().decode('utf-8-sig')))
            fields=[key.strip() for key in reader.fieldnames or []];reader.fieldnames=fields
            if len(fields)!=len(set(fields)) or not set(source.FEATURES+['loan_term']).issubset(fields):raise ValueError('Use the small-loan sample CSV headers, including loan_term for arithmetic only.')
            rows=[]
            for line,row in enumerate(reader,2):
                if len(rows)>=200:raise ValueError('Maximum 200 applications per upload.')
                try:rows.append(source.validate_personal(row))
                except ValueError as error:raise ValueError(f'CSV line {line}: {error}') from None
            if not rows:raise ValueError('CSV contains no applications.')
            outputs=source.run_personal(models,rows)
            ids=save_batch(app.config['DATABASE'],list(zip(rows,outputs)),report['best_model'],source.VERSION)
            return redirect(url_for('batch_personal',receipt=receipts.dumps(ids)),code=303)
        except (UnicodeError,csv.Error):return render_template('personal_batch.html',page='batch',error='Use a valid UTF-8 CSV.'),400
        except ValueError as error:return render_template('personal_batch.html',page='batch',error=str(error)),400
        except DATABASE_ERRORS:
            app.logger.error('Small-loan batch could not be saved')
            return render_template('personal_batch.html',page='batch',error='Upload could not be saved. No partial batch was kept.'),503

    @app.get('/sample/personal.csv')
    def sample_personal():
        rows=[source.EXAMPLE,{**source.EXAMPLE,'person_income':240000,'loan_amnt':20000,'previous_loan_defaults_on_file':'Yes'}]
        return csv_download(rows,source.FEATURES+['loan_term'],'loanlens-small-loan-sample.csv')

    @app.get('/dataset/personal.csv')
    def personal_dataset():
        return send_file(source.ROOT/'data/personal_loan_data.csv',as_attachment=True,download_name='loan_data.csv')
