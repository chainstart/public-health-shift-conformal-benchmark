"""Retain all eight submitted budgets using the corrected, fixed-role runner."""
import hashlib,json,sys
from pathlib import Path

interface=Path(__file__).parent/'benchmark'
if not interface.exists():interface=Path(__file__).parent/'reproduction/benchmark'
sys.path.insert(0,str(interface))
import run_budget

if __name__=='__main__':
    run_budget.FRACTIONS=(.005,.01,.02,.03,.05,.10,.15,.20)
    # main() parses the documented --lab-root / --out arguments.
    run_budget.main()
    out=Path(sys.argv[sys.argv.index('--out')+1])
    (out/'eight-budget-protocol.json').write_text(json.dumps({
        'reason':'Preserve all eight target fractions in the actual submitted manuscript.',
        'fractions':run_budget.FRACTIONS,'replicates':5,'models':60,
        'runner_sha256':hashlib.sha256(Path(run_budget.__file__).read_bytes()).hexdigest(),
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'uses_test_labels_for_fitting':False},indent=2)+'\n')
