import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from benchmark.api import MethodSpec, ThresholdSets, evaluate_method, finite_quantile, set_metrics

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkContractTests(unittest.TestCase):
    def test_preparation_stages_self_contained_core(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from benchmark.__main__ import prepare
        with tempfile.TemporaryDirectory() as folder:
            workspace=Path(folder)/'fresh'
            with patch('benchmark.__main__.subprocess.run') as run:
                prepare(SimpleNamespace(workspace=workspace,provider_cache=None))
            self.assertEqual(run.call_count,2)
            for relative in ('benchmark/prepare_data.py','scripts/run_public_shift_benchmark_v2.py',
                    'scripts/run_large_scale_conformal.py',
                    'exp/row_level_remediation_20260725/generate_row_level_predictions.py',
                    'exp/source_role_outcome_blind_20260725/generate.py'):
                self.assertTrue((workspace/relative).is_file())
            with self.assertRaises(ValueError):
                prepare(SimpleNamespace(workspace=workspace,provider_cache=None))

    def test_overflow_and_ties(self):
        self.assertEqual(finite_quantile(np.arange(8)/10), float('inf'))
        self.assertEqual(finite_quantile(np.array([.5]*20)), .5)
        self.assertTrue(ThresholdSets(.5,.5).predict_sets(np.array([.5])).all())

    def test_test_labels_are_not_exposed(self):
        calls=[]
        def fit(data):
            self.assertIsNone(data.target_labels)
            self.assertEqual(len(data.source_labels),20)
            self.assertFalse(hasattr(data,'test_labels'))
            calls.append(True)
            return ThresholdSets(.5,.5)
        result=evaluate_method(MethodSpec('probe',False,fit),np.full(20,.5),np.zeros(20,int),
            np.full(20,.5),np.ones(20,int),np.array([.2,.8]),np.array([0,1]))
        self.assertEqual(result['coverage'],1)
        self.assertEqual(calls,[True])

    def test_invalid_membership_rejected(self):
        class Bad:
            def predict_sets(self,p): return np.ones((len(p),2),dtype=int)
        with self.assertRaises(ValueError):
            evaluate_method(MethodSpec('bad',False,lambda d:Bad()),np.array([.2]),np.array([0]),
                np.array([.2]),np.array([0]),np.array([.2]),np.array([0]))

    def test_metrics_by_hand(self):
        m=set_metrics(np.array([[True,False],[True,True],[False,False],[False,True]]),[0,0,1,1])
        self.assertEqual(m['coverage'],.75)
        self.assertEqual(m['mean_set_size'],1)
        self.assertEqual(m['class_coverage_gap'],.5)
        self.assertEqual(m['empty_set_rate'],.25)

    def test_pairing_budget_and_role_overlap_rejected(self):
        def run(*args,ok=True):
            result=subprocess.run([sys.executable,'-m','benchmark',*map(str,args)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode == 0,ok,result.stderr)
        with tempfile.TemporaryDirectory() as folder:
            w=Path(folder)
            run('fixture','--workspace',w)
            run('evaluate','--workspace',w,'--methods','source_split','target_split','--out',w/'results.csv')
            frame=pd.read_csv(w/'results.csv')
            frame.loc[frame.method=='target_split','source_n']=301
            frame.to_csv(w/'bad.csv',index=False)
            run('compare','--input',w/'bad.csv','--out',w/'comparison',ok=False)
            path=next((w/'fixture').glob('*.parquet'))
            frame=pd.read_parquet(path);frame.loc[1,'row_id']=frame.loc[0,'row_id'];frame.to_parquet(path,index=False)
            import hashlib
            manifest=w/'exp/source_role_outcome_blind_20260725/generation_provenance.json'
            data=json.loads(manifest.read_text())
            for pair in data['pairs']:
                if pair['prediction_path']==str(path.relative_to(w)):
                    pair['prediction_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            run('evaluate','--workspace',w,'--out',w/'overlap.csv',ok=False)

    def test_end_to_end_and_input_tampering(self):
        def run(*args,ok=True):
            p=subprocess.run([sys.executable,'-m','benchmark',*map(str,args)],cwd=ROOT,capture_output=True,text=True)
            if ok: self.assertEqual(p.returncode,0,p.stderr)
            else: self.assertNotEqual(p.returncode,0)
        with tempfile.TemporaryDirectory() as folder:
            w=Path(folder);out=w/'results.csv'
            run('fixture','--workspace',w)
            run('evaluate','--workspace',w,'--methods','source_split','source_mondrian',
                'target_split','target_mondrian','examples/source_split.py','--out',out)
            frame=pd.read_csv(out)
            self.assertEqual(len(frame),50)
            a=frame[frame.method=='source_split'].drop(columns='method').reset_index(drop=True)
            b=frame[frame.method=='example_source_split'].drop(columns='method').reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)
            run('compare','--input',out,'--out',w/'comparison')
            self.assertEqual(len(pd.read_csv(w/'comparison/equal_dataset_summary.csv')),5)
            receipt=json.loads(out.with_suffix('.receipt.json').read_text())
            self.assertTrue(receipt['synthetic'])
            frame.iloc[:-1].to_csv(w/'incomplete.csv',index=False)
            run('compare','--input',w/'incomplete.csv','--out',w/'bad',ok=False)
            path=next((w/'fixture').glob('*.parquet'));path.write_bytes(path.read_bytes()+b'tamper')
            run('evaluate','--workspace',w,'--out',w/'bad.csv',ok=False)


if __name__=='__main__': unittest.main()
