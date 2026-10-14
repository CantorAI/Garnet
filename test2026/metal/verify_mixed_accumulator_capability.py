"""Independent archive and two distinguishing accumulator-fixture audit."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'manifest.json').read_text())
for name,info in proof['files'].items():
    p=root/name
    assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
folder=root/'logs/mixed-accumulator-capability-20261015';jobs=folder/'jobs'
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
stages=json.loads((jobs/'statuses.json').read_text())
assert [s['name'] for s in stages]==['compile','capabilities'] and all(s['exit_code']==0 for s in stages)
for s in stages:assert json.loads((jobs/(s['name']+'-monitor.json')).read_text())
source=(jobs/'source.mm').read_bytes()
assert source==(root/'source/test2026/metal/mixed_accumulator_capability_probe.mm').read_bytes()
assert b'simdgroup_float8x8 c=' in source and b'options.fastMathEnabled=NO' in source
assert b'for(size_t i=0;i<64;++i)if(actual[i]!=expected' in source
result=json.loads((jobs/'capabilities.log').read_text())
assert result['supported'] and result['device']=='Apple M4' and result['exact_outputs']==128
fixtures=result['fp32_accumulator_fixtures'];assert len(fixtures)==2
for fixture,expected_initial,expected_value,expected_half in zip(fixtures,(2048,0),(.25,2**-14),(2048,0)):
    expected=expected_initial+1024*expected_value**2
    assert fixture['initial']==expected_initial and fixture['half_input']==expected_value
    assert fixture['fp32_actual']==fixture['fp32_expected']==expected and fixture['exact_outputs']==64
    assert fixture['half_accumulator_actual']==expected_half and expected!=expected_half
audit=dict(verified_files=len(proof['files']),exact_outputs=128,distinguishing_fixtures=2,supported=True,scope=result['scope'],full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(audit,indent=2)+'\n').encode());print(json.dumps(audit))
