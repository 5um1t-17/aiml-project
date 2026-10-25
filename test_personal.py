"""Cross-source coverage, training reproducibility and persistence regression checks."""
import csv,io,json,hashlib,tempfile,unittest
from pathlib import Path
import joblib,numpy as np
from app import app,models,metrics,personal_models,personal_metrics,screen_record
from database import init_database,get_application,get_history,dashboard_stats,connect
from demo_data import seed_demo_if_empty,reset_with_backup,refresh_unreviewed_demos
from contextlib import closing
import sqlite3
import personal_loans as source

class PersonalLoanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=source.ROOT,prefix='test-db-')
        self.original=app.config['DATABASE'];app.config['DATABASE']=Path(self.temp.name)/'test.db'
        init_database(app.config['DATABASE']);self.client=app.test_client()
    def tearDown(self):
        app.config['DATABASE']=self.original;self.temp.cleanup()
    def predict(self,**changes):
        response=self.client.post('/api/predict/personal',json={**source.EXAMPLE,**changes})
        self.assertEqual(response.status_code,200);return response.get_json()
    def test_all_source_rows_and_separate_holdout_metrics(self):
        from train_personal import load_personal,split
        from train import evaluate
        raw=load_personal();X,y,groups,train,test=split(raw)
        self.assertEqual(len(raw),45000)
        self.assertEqual(hashlib.sha256((source.ROOT/'data/personal_loan_data.csv').read_bytes()).hexdigest(),personal_metrics['dataset_sha256'])
        self.assertFalse(set(groups[train])&set(groups[test]))
        evaluation=joblib.load(source.ROOT/'models/personal-evaluation-pipelines.joblib');probabilities=[]
        for name,model in personal_models.items():
            self.assertEqual(int(model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].n_samples_seen_),45000)
            self.assertEqual(set(model.feature_names_in_),set(source.FEATURES))
            self.assertFalse({'person_age','person_gender','loan_status','loan_term'}.intersection(model.feature_names_in_))
            old=evaluation[name];self.assertEqual(int(old.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].n_samples_seen_),len(train))
            probability=old.predict_proba(X.iloc[test])[:,1];probabilities.append(probability)
            saved=next(row for row in personal_metrics['models'] if row['name']==name)
            self.assertAlmostEqual(evaluate(y.iloc[test],probability,old.predict(X.iloc[test]))['accuracy'],saved['accuracy'])
        self.assertAlmostEqual(evaluate(y.iloc[test],np.mean(probabilities,axis=0))['accuracy'],personal_metrics['ensemble']['accuracy'])
    def test_smaller_incomes_loans_and_equal_weight_mean(self):
        for changes in [dict(person_income=8000,loan_amnt=500),dict(person_income=20000,loan_amnt=10000),dict(person_income=240000,loan_amnt=20000)]:
            result=self.predict(**changes)
            self.assertFalse(result['screening']['requires_review'],result['screening']['flags'])
            self.assertEqual(len(result['predictions']),3)
            self.assertAlmostEqual(result['summary']['approval_score'],sum(row['probability'] for row in result['predictions'])/3)
            self.assertEqual(result['summary']['review_status'],'pending')
    def test_term_is_only_used_for_repayment_arithmetic(self):
        a=self.predict(loan_term=1);b=self.predict(loan_term=4)
        self.assertEqual(a['predictions'],b['predictions'])
        self.assertAlmostEqual(a['screening']['minimum_monthly_principal'],b['screening']['minimum_monthly_principal']*4)
        self.assertIn('excluded from model prediction',a['screening']['note'])
    def test_out_of_range_inputs_remain_flagged(self):
        result=self.predict(person_income=100,loan_amnt=50000)
        self.assertTrue(result['screening']['requires_review'])
        self.assertEqual(result['summary']['assessment'],'Manual review required')
        self.assertEqual(get_application(app.config['DATABASE'],result['application_id'])['reviews'],[])
    def test_validation_and_no_silent_source_conversion(self):
        for change in [dict(person_income='NaN'),dict(loan_amnt=-5),dict(person_emp_exp=125),dict(loan_term=2.5),dict(previous_loan_defaults_on_file='unknown')]:
            self.assertEqual(self.client.post('/api/predict/personal',json={**source.EXAMPLE,**change}).status_code,400)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],0)
        a=self.predict();b=self.predict(previous_loan_defaults_on_file='Yes')
        self.assertNotEqual(a['predictions'],b['predictions'])
        self.assertEqual(self.client.post('/api/predict',json=source.EXAMPLE).status_code,400)
    def test_history_export_detail_and_explanation_retain_source(self):
        result=self.predict(person_income=240000,loan_amnt=20000);key=result['application_id']
        row=get_application(app.config['DATABASE'],key)
        self.assertEqual(row['input_schema'],source.SCHEMA);self.assertEqual(row['model_version'],source.VERSION)
        self.assertEqual(json.loads(row['input_json'])['loan_amnt'],20000)
        self.assertIsNone(row['credit_history']);self.assertIsNone(row['dependents'])
        page=self.client.get(f'/applications/{key}').get_data(as_text=True)
        for text in ['Twenty thousand','Credit score','45,000 synthetic records','arithmetic only']:
            self.assertIn(text,page)
        self.assertNotIn('previous mortgage input format',page)
        exported=list(csv.DictReader(io.StringIO(self.client.get('/export.csv').get_data(as_text=True))))
        self.assertTrue(all(float(item['loan_amnt'])==20000 and item['loan_amount']=='' for item in exported))
        self.assertEqual(self.client.get('/history').status_code,200)
    def test_personal_pages_batch_and_atomic_validation(self):
        for url in ['/predict','/predict?dataset=assets','/comparison?dataset=personal','/batch/personal','/sample/personal.csv','/dataset/personal.csv']:
            with self.client.get(url) as response:self.assertEqual(response.status_code,200)
        data=self.client.get('/sample/personal.csv').data
        def upload(body):return self.client.post('/batch/personal',data={'file':(io.BytesIO(body),'sample.csv')},content_type='multipart/form-data',follow_redirects=True)
        response=upload(data);self.assertEqual(response.status_code,200)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],2)
        self.assertEqual(self.client.get(response.request.full_path).status_code,200)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],2)
        self.assertEqual(upload(data.replace(b'20000',b'-5',1)).status_code,400)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],2)

    def test_complete_project_defaults_cover_both_sources_and_workflows(self):
        ids=seed_demo_if_empty(app.config['DATABASE'],models,metrics,personal_models,personal_metrics)
        self.assertEqual(ids,[1,2,3,4])
        rows=[get_application(app.config['DATABASE'],key) for key in ids]
        self.assertTrue(all(row['is_demo'] and row['demo_label'] for row in rows))
        self.assertEqual(sum(row['input_schema']==source.SCHEMA for row in rows),2)
        self.assertEqual([row['review_status'] for row in rows],['accepted','rejected','pending','pending'])
        for row in rows:
            clean=json.loads(row['input_json'])
            if row['input_schema']==source.SCHEMA:expected=source.run_personal(personal_models,[clean])[0]
            else:
                from prediction import run_models
                expected=run_models(models,[clean])[0]
            self.assertEqual([p['probability'] for p in row['predictions']],[p['probability'] for p in expected])
            self.assertEqual(self.client.get(f'/applications/{row["id"]}').status_code,200)
        self.assertFalse(any(screen_record(row)['requires_review'] for row in rows[:3]))
        self.assertIn('principal_exceeds_income',[flag['code'] for flag in screen_record(rows[3])['flags']])
        self.assertTrue(all('SIMULATED DEMO REVIEW' in row['reviews'][0]['notes'] for row in rows[:2]))
        stats=dashboard_stats(app.config['DATABASE'])
        self.assertEqual((stats['total'],stats['accepted'],stats['rejected'],stats['pending']),(4,1,1,2))
        self.assertEqual(seed_demo_if_empty(app.config['DATABASE'],models,metrics,personal_models,personal_metrics),[])
        self.assertEqual(refresh_unreviewed_demos(app.config['DATABASE'],models,metrics)['new_demo_ids'],[])
        self.assertEqual(get_history(app.config['DATABASE'])['total'],4)

    def test_complete_demo_reset_backs_up_existing_records_and_reviews(self):
        seed_demo_if_empty(app.config['DATABASE'],models,metrics,personal_models,personal_metrics)
        self.predict()
        backup,ids=reset_with_backup(app.config['DATABASE'],Path(self.temp.name)/'backups',models,metrics,personal_models,personal_metrics)
        with closing(sqlite3.connect(backup)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM applications').fetchone()[0],5)
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM staff_reviews').fetchone()[0],2)
        self.assertEqual(ids,[1,2,3,4])
        self.assertEqual(get_history(app.config['DATABASE'])['total'],4)
        with closing(connect(app.config['DATABASE'])) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],12)
            self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(),[])

if __name__=='__main__':unittest.main()
