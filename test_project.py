"""Integration checks using fictional inputs and isolated SQLite databases."""
import csv,io,json,sqlite3,tempfile,unittest,hashlib
from contextlib import closing
from pathlib import Path
import pandas as pd
import joblib,numpy as np
from sklearn.model_selection import train_test_split
from app import app,models,metrics
from database import init_database,get_application,get_history,save_application,save_batch,dashboard_stats,connect,insert_application,record_review
from demo_data import DEMOS,seed_demo_if_empty,reset_with_backup,refresh_unreviewed_demos
from prediction import validate,run_models,assess_support,summarize_predictions,explain_linear
from loan_schema import FEATURES,SCHEMA_ID,DATASET_PATH
from amounts import amount_in_words, format_amount

class WebsiteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='test-db-')
        self.old_database=app.config['DATABASE'];app.config['DATABASE']=Path(self.temp.name)/'test.db'
        init_database(app.config['DATABASE']);self.client=app.test_client();self.sample=dict(DEMOS[0][1])
    def tearDown(self):
        app.config['DATABASE']=self.old_database;self.temp.cleanup()
    def new_application(self,**changes):
        response=self.client.post('/api/predict',json={**self.sample,**changes});self.assertEqual(response.status_code,200)
        return response.get_json()
    def upload(self,body):
        return self.client.post('/batch',data={'file':(io.BytesIO(body),'applications.csv')},content_type='multipart/form-data',follow_redirects=True)
    def export(self,query=''):
        return list(csv.DictReader(io.StringIO(self.client.get('/export.csv'+query).get_data(as_text=True))))
    def review_form(self,key,**changes):
        self.client.get(f'/applications/{key}')
        with self.client.session_transaction() as session:token=session['csrf_token']
        form=dict(csrf_token=token,expected_review_id='0',reviewer_name='Test Reviewer',reviewer_role='Employee',decision='accepted',notes='Fictional application reviewed independently in an isolated test.',confirmed='yes')
        form.update(changes);return form

    def test_pages_and_assets(self):
        for path in ['/','/predict','/batch','/comparison','/about','/history','/static/style.css','/static/workspace.css','/static/app.js','/static/amounts.js','/health','/dataset.csv']:
            with self.subTest(path=path),self.client.get(path) as response:self.assertEqual(response.status_code,200)
    def test_monetary_words_and_rounding(self):
        self.assertEqual(format_amount('1.005'),'1.01')
        examples = {0:'Zero', 20000:'Twenty thousand', 1200000:'Twelve lakh', 10000000:'One crore',
                    123456.05:'One lakh twenty three thousand four hundred fifty six point zero five',
                    '1.005':'One point zero one', '-0.004':'Zero', 100000000000:'Ten thousand crore'}
        for value,expected in examples.items():
            with self.subTest(value=value):self.assertEqual(amount_in_words(value),expected)
        for invalid in ['',None,'NaN','Infinity','not a number']:
            self.assertEqual(amount_in_words(invalid),'')
    def test_words_on_saved_amounts_without_changing_model_inputs(self):
        result=self.new_application(loan_amount=20000)
        for url in ['/','/history',f'/applications/{result["application_id"]}']:
            with self.subTest(url=url):self.assertIn('Twenty thousand',self.client.get(url).get_data(as_text=True))
        detail=self.client.get(f'/applications/{result["application_id"]}').get_data(as_text=True)
        self.assertIn('Twelve lakh',detail)
        record=get_application(app.config['DATABASE'],result['application_id'])
        self.assertEqual(json.loads(record['input_json'])['loan_amount'],20000)
    def test_training_source_checksum_and_schema(self):
        self.assertEqual(hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),metrics['dataset_sha256'])
        raw=pd.read_csv(DATASET_PATH);self.assertEqual(len(raw),4269)
        self.assertNotIn('loan_status',FEATURES);self.assertNotIn('loan_id',FEATURES)
        self.assertEqual(metrics['audit']['negative_asset_values']['residential_assets_value'],28)
    def test_all_row_models_and_separate_evaluation_artifact(self):
        from train import load_data,evaluate
        from loan_schema import ASSETS
        raw=load_data();X=raw[FEATURES].copy();y=raw.loan_status.map({'Rejected':0,'Approved':1})
        for field in ASSETS:X.loc[X[field]<0,field]=np.nan
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
        evaluation=joblib.load(Path(__file__).parent/metrics['evaluation_artifact'])
        self.assertEqual(metrics['production_train_rows'],len(raw))
        self.assertEqual(metrics['support']['train_rows'],len(raw))
        self.assertEqual(metrics['evaluation_support']['train_rows'],len(X_train))
        scores=[]
        for name,model in models.items():
            with self.subTest(model=name):
                prep=model.named_steps['preprocess']
                scaler=prep.named_transformers_['numeric'].named_steps['scale']
                self.assertEqual(int(scaler.n_samples_seen_),len(raw))
                self.assertFalse({'loan_id','loan_status'}.intersection(prep.feature_names_in_))
                pd.testing.assert_series_equal(model.named_steps['features'].medians_,X[model.named_steps['features'].columns_].median())
                old=evaluation[name]
                self.assertEqual(int(old.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].n_samples_seen_),len(X_train))
                probability=old.predict_proba(X_test)[:,1];scores.append(probability)
                actual=evaluate(y_test,probability,old.predict(X_test));saved=next(row for row in metrics['models'] if row['name']==name)
                for key in ['accuracy','balanced_accuracy','f1','auc','brier']:
                    self.assertAlmostEqual(actual[key],saved[key])
        self.assertAlmostEqual(evaluate(y_test,np.mean(scores,axis=0))['accuracy'],metrics['ensemble']['accuracy'])
        self.assertIn('Trained on all 4269 source rows',self.client.get('/comparison').get_data(as_text=True))
    def test_refitted_predictions_get_new_version_and_legacy_scores_survive(self):
        clean=validate(self.sample)
        old_scores=[dict(name=name,approved=True,probability=.7) for name in models]
        old_id=save_application(app.config['DATABASE'],clean,old_scores,'Decision Tree',metrics['evaluation_model_version'])
        new_id=self.new_application()['application_id']
        old=get_application(app.config['DATABASE'],old_id);new=get_application(app.config['DATABASE'],new_id)
        self.assertEqual(old['model_version'],metrics['evaluation_model_version'])
        self.assertEqual(new['model_version'],metrics['model_version'])
        self.assertNotEqual(old['model_version'],new['model_version'])
        self.assertTrue(all(row['probability']==.7 for row in old['predictions']))
    def test_full_amount_and_annual_income_are_not_rescaled_for_model(self):
        example={**self.sample,'loan_amount':20000,'income_annum':1200000}
        normalized=validate(example)
        self.assertEqual(normalized['loan_amount'],20000);self.assertEqual(normalized['income_annum'],1200000)
        actual=self.new_application(loan_amount=20000)
        self.assertEqual(actual['predictions'],run_models(models,[normalized])[0])
    def test_three_models_and_average(self):
        result=self.new_application();self.assertEqual(len(result['predictions']),3)
        self.assertAlmostEqual(result['summary']['approval_score'],sum(row['probability'] for row in result['predictions'])/3)
        self.assertEqual(result['summary']['review_status'],'pending')
        for model in result['predictions']:self.assertTrue(0<=model['probability']<=1)
    def test_invalid_inputs_save_nothing(self):
        invalid=[{},[],None,{**self.sample,'income_annum':0},{**self.sample,'income_annum':'NaN'},{**self.sample,'loan_amount':'inf'},
            {**self.sample,'cibil_score':1},{**self.sample,'cibil_score':901},{**self.sample,'residential_assets_value':-1},
            {**self.sample,'no_of_dependents':1.5},{**self.sample,'loan_term':2.5},{**self.sample,'education':'unknown'},
            {**self.sample,'self_employed':'unknown'},{**self.sample,'income_annum':True},
            dict(ApplicantIncome=5000,CoapplicantIncome=1500,LoanAmount=120)]
        for data in invalid:
            with self.subTest(data=data):self.assertEqual(self.client.post('/api/predict',json=data).status_code,400)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],0)
    def test_source_fields_persist_and_uncollected_fields_are_null(self):
        result=self.new_application();key=result['application_id'];init_database(app.config['DATABASE'])
        row=get_application(app.config['DATABASE'],key)
        self.assertEqual(row['input_schema'],SCHEMA_ID);self.assertEqual(json.loads(row['input_json']),validate(self.sample))
        for field in ['credit_history','coapplicant_income','property_area']:self.assertIsNone(row[field])
        page=self.client.get(f'/applications/{key}').get_data(as_text=True)
        self.assertIn('Annual income',page);self.assertIn('CIBIL score',page)
        self.assertIn('8 years (project assumption)',page)
    def test_score_changes_with_real_cibil_input(self):
        a=self.new_application(cibil_score=740);b=self.new_application(cibil_score=400)
        self.assertTrue(any(abs(x['probability']-y['probability'])>.01 for x,y in zip(a['predictions'],b['predictions'])))
    def test_repayment_and_out_of_support_case(self):
        result=self.new_application(**DEMOS[2][1]);checks=result['screening']
        self.assertEqual(result['summary']['assessment'],'Manual review required')
        self.assertAlmostEqual(checks['monthly_income'],6000/12)
        self.assertAlmostEqual(checks['minimum_monthly_principal'],20000/12)
        self.assertTrue({'principal_exceeds_income','income_outside_training','sparse_term'}.issubset({flag['code'] for flag in checks['flags']}))
        self.assertEqual(get_application(app.config['DATABASE'],result['application_id'])['reviews'],[])
    def test_flags_do_not_manipulate_scores(self):
        flags=assess_support(validate(DEMOS[2][1]),metrics['support'])
        result=summarize_predictions([dict(probability=.9),dict(probability=.8),dict(probability=.85)],flags)
        self.assertEqual(result['assessment'],'Manual review required');self.assertAlmostEqual(result['approval_score'],.85)
    def test_unknown_support_requires_review(self):
        self.assertTrue(assess_support(validate(self.sample),None)['requires_review'])
    def test_mean_is_not_majority_vote(self):
        summary=summarize_predictions([dict(probability=.51),dict(probability=.51),dict(probability=.01)])
        self.assertAlmostEqual(summary['approval_score'],1.03/3);self.assertEqual(summary['assessment'],'Decline indication')
    def test_new_applications_do_not_record_staff_decisions(self):
        self.new_application();stats=dashboard_stats(app.config['DATABASE'])
        self.assertEqual((stats['pending'],stats['accepted'],stats['rejected']),(1,0,0))
    def test_batch_and_export_preserve_source_units(self):
        sample=self.client.get('/sample.csv').data;self.assertEqual(self.upload(sample).status_code,200)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],4)
        rows=self.export();self.assertEqual(len(rows),12)
        first=[row for row in rows if row['id']=='1'][0]
        self.assertEqual(float(first['income_annum']),1200000);self.assertEqual(float(first['loan_amount']),3000000)
        self.assertEqual(float(first['cibil_score']),740)
    def test_batch_strips_original_source_whitespace(self):
        body=self.client.get('/sample.csv').get_data(as_text=True)
        lines=body.splitlines();lines[0]=', '.join(lines[0].split(','))
        self.assertEqual(self.upload(('\n'.join(lines)+'\n').encode()).status_code,200)
    def test_bad_batch_is_atomic(self):
        sample=self.client.get('/sample.csv').data
        broken=[b'wrong,column\n1,2',sample.replace(b'1200000',b'-1',1),sample.splitlines()[0]+b'\n',b'\xff\xfe',sample.splitlines()[0]+b'\n'+b'\n'.join([sample.splitlines()[1]]*201)]
        for body in broken:
            with self.subTest(body=body[:40]):
                self.assertEqual(self.upload(body).status_code,400);self.assertEqual(get_history(app.config['DATABASE'])['total'],0)
    def test_batch_database_failure_rolls_back(self):
        good=[dict(name='one',approved=True,probability=.5)]
        with self.assertRaises(sqlite3.IntegrityError):save_batch(app.config['DATABASE'],[(validate(self.sample),good),(validate(self.sample),good+good)],'one')
        self.assertEqual(get_history(app.config['DATABASE'])['total'],0)
    def test_receipt_refresh_does_not_duplicate(self):
        result=self.upload(self.client.get('/sample.csv').data)
        self.assertEqual(self.client.get(result.request.path+'?'+result.request.query_string.decode()).status_code,200)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],4)
        self.assertEqual(self.client.get('/batch?receipt=fake').status_code,400)
    def test_pagination_and_search(self):
        self.assertIn('No saved applications yet',self.client.get('/history').get_data(as_text=True))
        predictions=run_models(models,[validate(self.sample)])[0]
        for _ in range(21):save_application(app.config['DATABASE'],validate(self.sample),predictions,metrics['best_model'],SCHEMA_ID)
        self.assertEqual(len(get_history(app.config['DATABASE'],page=2)['records']),1)
        for value in ['0','2','-1','999','invalid']:self.assertEqual(self.client.get('/history?page='+value).status_code,200)
        self.assertEqual(len(self.export('?q=1')),3)
        self.assertEqual(self.client.get('/applications/99999').status_code,404)
    def test_linear_explanation_reconstructs_score(self):
        clean=validate(self.sample);model=models['Logistic Regression'];explanation=explain_linear(model,clean)
        value=sum(row['contribution'] for row in explanation)+model.named_steps['classifier'].intercept_[0]
        self.assertAlmostEqual(value,float(model.decision_function(pd.DataFrame([clean]))[0]))
        self.assertEqual(len({row['label'] for row in explanation}),len(explanation))
    def test_explicit_review_retains_audit_and_model_scores(self):
        result=self.new_application();key=result['application_id'];url=f'/applications/{key}/review';form=self.review_form(key)
        self.assertEqual(self.client.post(url,data=form,follow_redirects=True).status_code,200)
        row=get_application(app.config['DATABASE'],key);self.assertEqual(row['review_status'],'accepted')
        form=self.review_form(key,decision='rejected',reviewer_role='Manager',expected_review_id=str(row['reviews'][0]['id']))
        self.assertEqual(self.client.post(url,data=form).status_code,303)
        row=get_application(app.config['DATABASE'],key);self.assertEqual(row['review_status'],'rejected');self.assertEqual(len(row['reviews']),2)
        self.assertEqual([p['probability'] for p in row['predictions']],[p['probability'] for p in result['predictions']])
        self.assertEqual(len(self.export('?status=rejected')),3)
    def test_review_requires_csrf_confirmation_and_notes(self):
        key=self.new_application()['application_id'];form=self.review_form(key)
        for change in [dict(csrf_token=''),dict(confirmed=''),dict(notes=''),dict(reviewer_name=''),dict(reviewer_role='Admin'),dict(decision='automatic')]:
            self.assertEqual(self.client.post(f'/applications/{key}/review',data={**form,**change}).status_code,400)
        self.assertEqual(get_application(app.config['DATABASE'],key)['reviews'],[])
    def test_stale_review_cannot_overwrite(self):
        key=self.new_application()['application_id'];form=self.review_form(key);url=f'/applications/{key}/review'
        self.assertEqual(self.client.post(url,data=form).status_code,303)
        self.assertEqual(self.client.post(url,data={**form,'decision':'rejected'}).status_code,400)
        self.assertEqual(get_application(app.config['DATABASE'],key)['review_status'],'accepted')
    def test_csv_formula_escape(self):
        key=self.new_application()['application_id'];self.client.post(f'/applications/{key}/review',data=self.review_form(key,notes='=A1+1 Fictional spreadsheet test note.'))
        self.assertTrue(all(row['review_notes'].startswith("'=") for row in self.export()))
    def test_cross_site_posts_rejected(self):
        self.assertEqual(self.client.post('/api/predict',json=self.sample,headers={'Origin':'https://elsewhere.example'}).status_code,403)
    def test_database_errors(self):
        app.config['DATABASE']=Path(self.temp.name)
        with self.assertLogs(app.logger,level='ERROR'):self.assertEqual(self.client.post('/api/predict',json=self.sample).status_code,503)
        with self.assertLogs(app.logger,level='ERROR'):self.assertEqual(self.client.get('/history').status_code,503)
    def test_default_demos_are_labeled_and_not_duplicated(self):
        self.assertEqual(len(seed_demo_if_empty(app.config['DATABASE'],models,metrics)),4)
        rows=get_history(app.config['DATABASE'])['records'];self.assertTrue(all(row['is_demo'] and row['demo_label'] for row in rows))
        self.assertEqual(seed_demo_if_empty(app.config['DATABASE'],models,metrics),[])
        self.new_application();seed_demo_if_empty(app.config['DATABASE'],models,metrics)
        self.assertEqual(get_history(app.config['DATABASE'])['total'],5)
    def test_demo_reset_preserves_backup(self):
        key=self.new_application()['application_id'];self.client.post(f'/applications/{key}/review',data=self.review_form(key))
        backup,ids=reset_with_backup(app.config['DATABASE'],Path(self.temp.name)/'backups',models,metrics)
        with closing(sqlite3.connect(backup)) as old:
            self.assertEqual(old.execute('SELECT COUNT(*) FROM applications').fetchone()[0],1)
            self.assertEqual(old.execute('SELECT COUNT(*) FROM staff_reviews').fetchone()[0],1)
        self.assertEqual(ids,[1,2,3,4]);self.assertTrue(all(not get_application(app.config['DATABASE'],i)['reviews'] for i in ids))
    def test_legacy_migration_preserves_original_rows_and_scores(self):
        legacy=Path(self.temp.name)/'legacy.db'
        with closing(sqlite3.connect(legacy)) as old,old:
            old.executescript('''CREATE TABLE applications(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,applicant_income REAL NOT NULL,coapplicant_income REAL NOT NULL,loan_amount REAL NOT NULL,loan_term REAL NOT NULL,credit_history INTEGER NOT NULL,dependents TEXT NOT NULL,education TEXT NOT NULL,self_employed TEXT NOT NULL,property_area TEXT NOT NULL,selected_model TEXT NOT NULL);
            CREATE TABLE predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,application_id INTEGER REFERENCES applications(id),model_name TEXT NOT NULL,approved INTEGER NOT NULL,probability REAL NOT NULL,UNIQUE(application_id,model_name));''')
            old.execute("INSERT INTO applications VALUES(1,'old UTC',5000,1500,120,360,1,'0','Graduate','No','Urban','Logistic Regression')")
            for name in models:old.execute('INSERT INTO predictions(application_id,model_name,approved,probability) VALUES(1,?,1,.7)',(name,))
        init_database(legacy);app.config['DATABASE']=legacy
        row=get_application(legacy,1);self.assertEqual(row['applicant_income'],5000);self.assertEqual(row['credit_history'],1)
        self.assertEqual(row['input_schema'],'legacy-v1');self.assertEqual(len(row['predictions']),3)
        page=self.client.get('/applications/1').get_data(as_text=True);self.assertIn('previous mortgage dataset',page)
        self.assertIn('Manual review required',page)
        new=self.new_application();self.assertGreater(new['application_id'],1)
        self.assertIsNone(get_application(legacy,new['application_id'])['credit_history'])
        with closing(sqlite3.connect(legacy)) as connection:self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_refresh_updates_only_unreviewed_legacy_demos(self):
        old=dict(ApplicantIncome=5000,CoapplicantIncome=1500,LoanAmount=120,Loan_Amount_Term=360,Credit_History=1,Dependents='0',Education='Graduate',Self_Employed='No',Property_Area='Urban')
        scores=[dict(name=name,approved=True,probability=.7) for name in models]
        with closing(connect(app.config['DATABASE'])) as connection,connection:
            obsolete=insert_application(connection,old,scores,'Logistic Regression','old','Old demo')
            normal=insert_application(connection,old,scores,'Logistic Regression','old')
            reviewed=insert_application(connection,old,scores,'Logistic Regression','old','Reviewed old demo')
        record_review(app.config['DATABASE'],reviewed,'Test Reviewer','Employee','pending','Fictional preserved review entry.',0)
        result=refresh_unreviewed_demos(app.config['DATABASE'],models,metrics)
        self.assertEqual(result['replaced_demo_ids'],[obsolete]);self.assertEqual(len(result['new_demo_ids']),4)
        self.assertIsNotNone(get_application(app.config['DATABASE'],normal))
        self.assertEqual(len(get_application(app.config['DATABASE'],reviewed)['reviews']),1)
        self.assertEqual(refresh_unreviewed_demos(app.config['DATABASE'],models,metrics)['new_demo_ids'],[])

if __name__=='__main__':unittest.main()
