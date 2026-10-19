"""Acquire exact provider inputs, reject HTML payloads, and prepare frozen cohorts."""
import argparse,csv,hashlib,json,sys,urllib.request,zipfile
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def valid_header(raw,kind):
    return raw.startswith(b'PK') if kind=='zip' else raw.startswith(b'HEADER RECORD')

def inputs(root):
    raw=root/'data/raw_large';public=raw/'public_benchmark_v2';out=[]
    for year in [2011,2013,2015,2017,2019,2021,2023]:
        out.append(('BRFSS',str(year),f'LLCP{year}XPT.zip',f'https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip',public/f'brfss_{year}_xpt.zip','zip'))
    for year,suffix in [(2011,'G'),(2013,'H'),(2015,'I'),(2017,'J')]:
        for stem in ['DEMO','DIQ','BMX','BPQ','SMQ','MCQ']:
            name=f'{stem}_{suffix}.xpt'
            out.append(('NHANES',f'{year}-{year+1}',name,f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}',public/f'nhanes_{year}_{stem}_{suffix}.xpt','xpt'))
    out.append(('UCI','1999-2008','diabetic_data.csv',
                'https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip',
                raw/'uci_diabetes_130_hospitals.zip','zip'))
    return out

def probe(item):
    dataset,release,identifier,url,path,kind=item
    record=dict(dataset=dataset,release=release,file_identifier=identifier,url=url)
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'OpenLabs-public-data-reproduction/1.0','Range':'bytes=0-159'})
        with urllib.request.urlopen(req,timeout=35) as response:
            header=response.read(160);record.update(status=response.status,resolved_url=response.geturl(),
                                                    content_type=response.headers.get('Content-Type'),valid_file_signature=valid_header(header,kind))
    except Exception as exc:record.update(error=type(exc).__name__,valid_file_signature=False)
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lab-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--probe-urls',action='store_true')
    ap.add_argument('--download',action='store_true');ap.add_argument('--rebuild-cohorts',action='store_true')
    a=ap.parse_args();root=a.lab_root.resolve();a.out.mkdir(parents=True,exist_ok=True);items=inputs(root)
    for directory in ('data/processed_large', 'data/processed_large/public_benchmark_v2'):
        (root/directory).mkdir(parents=True,exist_ok=True)
    if a.probe_urls:
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(probe,items))
        record={'checked_at':datetime.now(timezone.utc).isoformat(),'files':results}
        (a.out/'remote-source-checks.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Valid remote file signatures:',sum(r['valid_file_signature'] for r in results),'/',len(results),flush=True)
    receipts=[]
    for dataset,release,identifier,url,path,kind in items:
        if not path.exists():
            if not a.download:raise FileNotFoundError(f'{identifier}: use --download to acquire {url}')
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.partial')
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'OpenLabs-public-data-reproduction/1.0'})
                with urllib.request.urlopen(req,timeout=180) as response,temporary.open('wb') as f:
                    first=response.read(1024*1024)
                    if not valid_header(first,kind):raise ValueError(f'Provider returned a non-{kind} payload for {identifier}')
                    f.write(first)
                    for block in iter(lambda:response.read(1024*1024),b''):f.write(block)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with path.open('rb') as f:assert valid_header(f.read(160),kind),identifier
        if kind=='zip':
            with zipfile.ZipFile(path) as z:
                assert z.namelist(),identifier
                if dataset=='UCI':assert 'diabetic_data.csv' in z.namelist()
        receipts.append(dict(dataset=dataset,release=release,file_identifier=identifier,url=url,
                             local_path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest(path)))
    (a.out/'provider-inputs.json').write_text(json.dumps(receipts,indent=2)+'\n')
    sys.path.insert(0,str(root))
    from scripts.run_public_shift_benchmark_v2 import prepare_brfss_multiyear,prepare_nhanes_multicycle,prepare_uci_with_pseudo
    import numpy as np
    import pandas as pd
    constructors=[lambda:prepare_brfss_multiyear(force=a.rebuild_cohorts,max_rows_per_year=60000),
                  lambda:prepare_nhanes_multicycle(force=a.rebuild_cohorts),lambda:prepare_uci_with_pseudo(force=a.rebuild_cohorts)]
    cohorts=[];design=[];features=[]
    for constructor in constructors:
        ds=constructor();cohorts.append({'dataset':ds.name,'records':ds.final_n,'outcome_prevalence':float(ds.y.mean())})
        features.append({'dataset':ds.name,'numeric_outcome_features':ds.numeric_cols,'categorical_outcome_features':ds.categorical_cols})
        for shift in ds.shifts:
            if shift.name=='pseudo_random':continue
            s=ds.groups[shift.group_col].isin(shift.source_values).to_numpy();t=ds.groups[shift.group_col].isin(shift.target_values).to_numpy()
            assert not np.any(s&t)
            description=shift.description
            if ds.name=='nhanes_multicycle_diabetes' and shift.name=='age_65plus':description='all participants younger than 65 versus age 65 and older; source includes children'
            if shift.name=='obesity':description='recorded BMI >=30 or BRFSS BMI category 4 versus other records, including unavailable BMI'
            design.append(dict(dataset=ds.name,shift_name=shift.name,source_n=int(s.sum()),target_n=int(t.sum()),
                               source_prevalence=float(ds.y[s].mean()),target_prevalence=float(ds.y[t].mean()),
                               group_col=shift.group_col,source_values='|'.join(map(str,shift.source_values)),
                               target_values='|'.join(map(str,shift.target_values)),description=description))
    target=root/'benchmark_release/public_health_shift_conformal_benchmark_v3/results/public_benchmark_shift_audit.csv'
    target.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(design).to_csv(target,index=False)
    pd.DataFrame(design).to_csv(a.out/'cohort_shift_definitions.csv',index=False)
    assert sum(c['records'] for c in cohorts)==528458
    (a.out/'cohorts.json').write_text(json.dumps(cohorts,indent=2)+'\n')
    (a.out/'outcome-model-features.json').write_text(json.dumps(features,indent=2)+'\n')
    print('Verified',len(receipts),'provider inputs and',len(design),'conditions; cohorts:',cohorts,flush=True)

if __name__=='__main__':main()
