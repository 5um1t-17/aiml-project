"""Regression checks for split leakage guards and completed retraining metadata."""
import json,unittest
from datetime import datetime
from pathlib import Path
import pandas as pd
from training_audit import checked_split


class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.X=pd.DataFrame({'income':[10,20,30,40,50,60]})
        self.y=pd.Series([0,1,0,1,0,1])
    def test_overlapping_or_incomplete_splits_are_rejected(self):
        for train,test in [([0,1,2,3,4],[4,5]),([0,1],[4,5]),([0,0,1,2,3],[4,5])]:
            with self.subTest(train=train,test=test),self.assertRaises(ValueError):
                checked_split(self.X,self.y,train,test)
    def test_duplicate_input_leakage_and_target_features_are_rejected(self):
        duplicate=self.X.copy();duplicate.loc[4,'income']=duplicate.loc[0,'income']
        for frame in [duplicate,self.X.assign(loan_status=self.y),self.X.assign(loan_id=range(6))]:
            with self.subTest(columns=list(frame)),self.assertRaises(ValueError):
                checked_split(frame,self.y,[0,1,2,3],[4,5])
    def test_saved_bundles_record_successful_split_and_fit_checks(self):
        root=Path(__file__).parent
        for file in ['kaggle-metrics.json','personal-metrics.json']:
            report=json.loads((root/'models'/file).read_text(encoding='utf-8'));run=report['training_run']
            with self.subTest(file=file):
                self.assertIsNotNone(datetime.fromisoformat(run['completed_at_utc']).tzinfo)
                self.assertEqual(run['split']['source_rows'],report['rows'])
                self.assertEqual(run['split']['overlapping_row_indices'],0)
                self.assertEqual(run['split']['identical_feature_groups_crossing_holdout'],0)
                self.assertTrue(run['convergence_warnings_are_errors'])
                self.assertEqual(len(run['fitted_models']),3)
                for fitted in run['fitted_models'].values():
                    self.assertEqual(fitted['evaluation_fit_rows'],report['train_rows'])
                    self.assertEqual(fitted['final_fit_rows'],report['rows'])

if __name__=='__main__':unittest.main()
