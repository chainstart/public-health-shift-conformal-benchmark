"""Reconstruct exact external role counts from saved prediction membership."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    receipt=json.loads((args.results/'run-receipt.json').read_text())
    assert not receipt['failures']
    rows=[]
    for item in receipt['inputs']:
        path=Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        frame=pd.read_parquet(path,columns=['row_id','role'])
        assert frame.row_id.is_unique
        counts=frame.groupby('role').size().to_dict()
        if item['condition']=='random_control':
            n=sum(counts.values())
            assert counts['target_test']==math.ceil(.25*n)
            assert counts['source_calibration']==math.ceil(.3*(n-counts['target_test']))
            assert counts['source_train']==n-counts['target_test']-counts['source_calibration']
        else:
            ns=counts['source_train']+counts['source_calibration']
            nt=counts['target_calibration_pool']+counts['target_test']
            assert counts['source_calibration']==math.ceil(.3*ns)
            assert counts['source_train']==ns-counts['source_calibration']
            assert counts['target_calibration_pool']==math.ceil(.3*nt)
            assert counts['target_test']==nt-counts['target_calibration_pool']
        rows.append(dict(dataset=item['dataset'],backbone=item['backbone'],condition=item['condition'],seed=item['seed'],**counts))
    fields=['dataset','backbone','condition','seed','source_train','source_calibration','target_calibration_pool','target_test']
    with (args.results/'role_counts.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result={'status':'pass','prediction_files_checked':len(rows),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'role_counts_sha256':hashlib.sha256((args.results/'role_counts.csv').read_bytes()).hexdigest(),
            'scope':'Exact external role rounding and row disjointness; no model or threshold fitting.'}
    (args.results/'role-rounding-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
