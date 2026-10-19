"""Replace only fully verified comparator rows; retain low-budget results."""
import argparse,json,shutil,hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--artifacts',type=Path,required=True);args=a.parse_args();root=Path(__file__).parent;out=root/'results';matched=out/'matched';low=out/'optimizer_low_budget';low.mkdir(exist_ok=True)
 art=args.artifacts;lr=art/'optimizer-full/lab/exp/lrqr_baseline_20260824';ot=art/'optimizer-full/ot'
 assert json.loads((lr/'validation_receipt.json').read_text())['status']=='PASS'
 assert json.loads((ot/'validation.json').read_text())['status']=='passed'
 if not (low/'extended_replicate_metrics.csv').exists():
  for p in matched.iterdir():
   if p.suffix in {'.csv','.json'}:shutil.copy2(p,low/p.name)
 keys=['dataset','shift','seed','replicate']
 base=pd.read_csv(low/'extended_replicate_metrics.csv',float_precision='round_trip');fresh=pd.read_parquet(lr/'lrqr_replicate_metrics.parquet');newot=pd.read_csv(ot/'replicate_metrics.csv',float_precision='round_trip');newot['method']='ot_minmax_weighted_cdf_300';newot['qhat_0']=newot.threshold;newot['qhat_1']=newot.threshold
 trace=pd.read_csv(ot/'optimization_trace.csv',float_precision='round_trip');trace=trace[trace.step==4000].set_index(keys)
 newot['diagnostics_json']=newot.apply(lambda row:json.dumps({'implementation':'independent_ot_minmax_weighted_cdf','steps':4000,'grid_points':401,'normalization':'calibration mean one; unit test mass','final_objective':float(trace.loc[tuple(row[k] for k in keys),'objective']),'logit_gradient_norm':float(trace.loc[tuple(row[k] for k in keys),'logit_gradient_norm'])}),axis=1)
 keys=['dataset','shift','seed','replicate'];metrics=['coverage','coverage_shortfall','mean_set_size','empty_set_rate','singleton_rate','full_set_rate','class_0_coverage','class_1_coverage','class_coverage_gap']
 for method,replacement in [('ot_minmax_weighted_cdf_300',newot),('lrqr_300_300_300',fresh)]:
  old=base[base.method==method].set_index(keys).sort_index();new=replacement.set_index(keys).sort_index();assert old.index.equals(new.index);assert len(new)==300
  for c in metrics+['qhat_0','qhat_1','diagnostics_json']:
   if c in new.columns:base.loc[base.method==method,c]=base.loc[base.method==method,keys].apply(lambda r:new.loc[tuple(r),c],axis=1).values
  for c in ['threshold_minimum','threshold_median','threshold_maximum','selected_lambda','selected_beta']:
   if c in new.columns:base.loc[base.method==method,c]=base.loc[base.method==method,keys].apply(lambda r:new.loc[tuple(r),c],axis=1).values
 assert len(base)==2700 and not base.duplicated(keys+['method']).any()
 assert np.allclose(base.singleton_rate+2*base.full_set_rate,base.mean_set_size)
 base.to_csv(matched/'extended_replicate_metrics.csv',index=False);fresh.to_csv(matched/'lrqr_replicate_metrics.csv',index=False)
 seeds=base.groupby(['dataset','shift','method','seed'])[metrics].mean();shift=seeds.groupby(['dataset','shift','method']).mean().reset_index();ds=shift.groupby(['dataset','method'])[metrics].mean().reset_index();eq=ds.groupby('method')[metrics].mean().reset_index()
 for name,x in [('extended_shift_summary.csv',shift),('extended_dataset_summary.csv',ds),('extended_equal_dataset_summary.csv',eq)]:x.to_csv(matched/name,index=False)
 for origin,name in [(lr,'optimizer_full_lrqr'),(ot,'optimizer_full_ot'),(art/'optimizer-stability/ot','optimizer_sensitivity_ot'),(art/'optimizer-stability/lrqr','optimizer_sensitivity_lrqr'),(art/'optimizer-stability/controls','optimizer_controls'),(art/'predictor-discrimination','predictor_discrimination')]:
  target=out/name;target.mkdir(exist_ok=True)
  for p in origin.iterdir():
   if p.suffix in {'.csv','.json'}:shutil.copy2(p,target/p.name)
 receipt={'status':'PASS','scope':'Only verified OT4000 and LRQR1800CV/8000final replace their prior300 rows; all seven remaining methods preserved. Independent verifier receipts copied, main aggregation rebuilt.','cases':600,'script_sha256':sha(Path(__file__)),'ot_verification_sha256':sha(ot/'validation.json'),'lrqr_verification_sha256':sha(lr/'validation_receipt.json'),'original_low_budget_sha256':sha(low/'extended_replicate_metrics.csv'),'updated_rows_sha256':sha(matched/'extended_replicate_metrics.csv')}
 (matched/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(matched/'results.json').write_text(json.dumps({'version':'1.4.2','methods':9,'method_rows':2700,'ot_steps':4000,'lrqr_cv_steps':1800,'lrqr_final_steps':8000,'scope':'Main numerical estimates replace low-budget comparator estimates; these are finite optimizations, not global-optimality certificates.','equal_dataset_summary':eq.to_dict('records')},indent=2)+'\n')
 print(eq[['method','coverage','mean_set_size','class_coverage_gap']].to_string(index=False))
if __name__=='__main__':main()
