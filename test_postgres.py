"""Opt-in PostgreSQL integration tests in a disposable, uniquely named schema."""
import csv,io,json,os,re,tempfile,unittest,uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import psycopg
from psycopg import sql
from app import app,models,metrics,personal_models,personal_metrics
from configuration import database_target,signing_keys
from database import connect,init_database,get_application,get_history,dashboard_stats,record_review,save_application,save_batch
from demo_data import DEMOS,seed_demo_if_empty
from prediction import validate,run_models
from personal_loans import EXAMPLE
from postgres_backend import is_postgres
from migrate_postgres import migrate


@unittest.skipUnless(os.environ.get('RUN_POSTGRES_TESTS')=='1','Opt in to isolated PostgreSQL integration tests.')
class PostgreSQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target=database_target()
        if not is_postgres(cls.target):raise unittest.SkipTest('DATABASE_URL is required.')
        cls.schema='loanlens_test_'+uuid.uuid4().hex
        cls.old_schema=os.environ.get('DATABASE_SCHEMA')
        cls.created=False
        with psycopg.connect(cls.target,autocommit=True,prepare_threshold=None,connect_timeout=10) as connection:
            connection.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(cls.schema)))
        cls.created=True
        os.environ['DATABASE_SCHEMA']=cls.schema
        init_database(cls.target)
    @classmethod
    def tearDownClass(cls):
        try:
            if cls.created and re.fullmatch(r'loanlens_test_[0-9a-f]{32}',cls.schema):
                with psycopg.connect(cls.target,autocommit=True,prepare_threshold=None,connect_timeout=10) as connection:
                    connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(cls.schema)))
        finally:
            if cls.old_schema is None:os.environ.pop('DATABASE_SCHEMA',None)
            else:os.environ['DATABASE_SCHEMA']=cls.old_schema
    def setUp(self):
        self.original=app.config['DATABASE'];app.config['DATABASE']=self.target
        self.client=app.test_client()
        with closing(connect(self.target)) as connection,connection:
            self.assertEqual(connection.execute('SELECT current_schema() AS name').fetchone()['name'],self.schema)
            connection.execute('TRUNCATE staff_reviews,predictions,applications RESTART IDENTITY CASCADE')
    def tearDown(self):app.config['DATABASE']=self.original
    def application(self):
        response=self.client.post('/api/predict',json=DEMOS[0][1])
        self.assertEqual(response.status_code,200);return response.get_json()
    def test_both_prediction_apis_history_filters_and_exports(self):
        original=self.application()
        personal=self.client.post('/api/predict/personal',json=EXAMPLE)
        self.assertEqual(personal.status_code,200)
        self.assertEqual(get_history(self.target)['total'],2)
        self.assertEqual(dashboard_stats(self.target)['pending'],2)
        for query in ['', '?status=approved','?status=declined','?status=pending','?q=graduate']:
            self.assertEqual(self.client.get('/history'+query).status_code,200)
            self.assertEqual(self.client.get('/export.csv'+query).status_code,200)
        exported=list(csv.DictReader(io.StringIO(self.client.get('/export.csv').get_data(as_text=True))))
        self.assertEqual(len(exported),6)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(self.client.get('/export.csv?q=graduate').get_data(as_text=True))))),3)
        for key in [original['application_id'],personal.json['application_id']]:self.assertEqual(self.client.get(f'/applications/{key}').status_code,200)
    def test_atomic_batch_rollback_and_foreign_keys(self):
        clean=validate(DEMOS[0][1]);scores=run_models(models,[clean])[0]
        with self.assertRaises(psycopg.IntegrityError):
            save_batch(self.target,[(clean,scores),(clean,scores+scores)],metrics['best_model'],metrics['model_version'])
        self.assertEqual(get_history(self.target)['total'],0)
        with self.assertRaises(psycopg.IntegrityError):
            with closing(connect(self.target)) as connection,connection:
                connection.execute("INSERT INTO predictions(application_id,model_name,approved,probability) VALUES(999,'missing',1,.5)")
    def test_concurrent_review_stale_check_and_original_scores(self):
        result=self.application();key=result['application_id']
        def submit(decision):
            try:
                record_review(self.target,key,'Integration Test Reviewer','Employee',decision,'Fictional application in an isolated test schema.',0)
                return 'saved'
            except ValueError:return 'stale'
        with ThreadPoolExecutor(max_workers=2) as executor:outcomes=list(executor.map(submit,['accepted','rejected']))
        self.assertEqual(sorted(outcomes),['saved','stale'])
        row=get_application(self.target,key);self.assertEqual(len(row['reviews']),1)
        self.assertEqual([p['probability'] for p in row['predictions']],[p['probability'] for p in result['predictions']])
    def test_sqlite_migration_idempotency_sequences_and_conflict_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='test-db-') as folder:
            source=Path(folder)/'source.db';init_database(source)
            seed_demo_if_empty(source,models,metrics,personal_models,personal_metrics)
            copied=migrate(source,self.target,Path(folder)/'backups')
            self.assertEqual(copied['counts'],dict(applications=4,predictions=12,staff_reviews=2))
            self.assertEqual(migrate(source,self.target,Path(folder)/'backups')['status'],'already_migrated')
            self.assertEqual(dashboard_stats(self.target)['accepted'],1)
            self.assertEqual(dashboard_stats(self.target)['rejected'],1)
            clean=validate(DEMOS[0][1]);scores=run_models(models,[clean])[0]
            self.assertEqual(save_application(self.target,clean,scores,metrics['best_model'],metrics['model_version']),5)
            with self.assertRaises(ValueError):migrate(source,self.target,Path(folder)/'backups')
            self.assertEqual(get_history(self.target)['total'],5)
            self.assertEqual(get_history(source)['total'],4)
    def test_both_csv_uploads_and_receipt_refresh(self):
        for sample,url in [('/sample.csv','/batch'),('/sample/personal.csv','/batch/personal')]:
            content=self.client.get(sample).data
            response=self.client.post(url,data={'file':(io.BytesIO(content),'fictional.csv')},content_type='multipart/form-data',follow_redirects=True)
            self.assertEqual(response.status_code,200)
            total=get_history(self.target)['total']
            self.assertEqual(self.client.get(response.request.full_path).status_code,200)
            self.assertEqual(get_history(self.target)['total'],total)
        self.assertEqual(get_history(self.target)['total'],6)
        self.assertEqual(signing_keys(),signing_keys())
    def test_review_route_csrf_and_stale_post(self):
        key=self.application()['application_id'];self.client.get(f'/applications/{key}')
        with self.client.session_transaction() as session:token=session['csrf_token']
        form=dict(csrf_token=token,expected_review_id='0',reviewer_name='Integration Test Reviewer',reviewer_role='Manager',decision='accepted',notes='Fictional review in a disposable PostgreSQL schema.',confirmed='yes')
        url=f'/applications/{key}/review'
        self.assertEqual(self.client.post(url,data={**form,'csrf_token':''}).status_code,400)
        self.assertEqual(self.client.post(url,data=form).status_code,303)
        self.assertEqual(self.client.post(url,data=form).status_code,400)
        self.assertEqual(get_application(self.target,key)['review_status'],'accepted')


if __name__=='__main__':unittest.main()
