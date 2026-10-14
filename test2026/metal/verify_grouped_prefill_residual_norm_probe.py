"""Verify each clean group independently; exclude all interrupted attempts."""
import json,subprocess,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();job=base/'evidence/logs/prefill-residual-norm-grouped-20261014'
assert json.loads((job/'exit.json').read_text())['exit_code']==0
outer=json.loads((job/'summary.json').read_text());groups=outer['groups'];assert outer['all_completed'] and len(groups)==12
assert {(g['mode'],g['index']) for g in groups}=={(m,i) for m in ('regular','extra') for i in range(6)}
reports=base/'confirmation/groups';reports.mkdir(parents=True,exist_ok=True);profiles=[];counts=0;windows=0;worst=0
for g in groups:
 name=g['evidence'];assert '/' not in name and '\\' not in name
 target=reports/(g['mode']+'-case'+str(g['index'])+'.json')
 subprocess.run([sys.executable,str(Path(__file__).with_name('verify_prefill_residual_norm_probe.py')),str(base),str(target),'prefill-residual-norm-grouped-20261014/'+name],check=True)
 result=json.loads(target.read_text());profiles.extend(result['profiles']);counts+=result['exact_validated_executions'];windows+=result['timing_windows'];worst=max(worst,result['fp64_worst_ratio'])
assert len(profiles)==132
summary=dict(profiles=profiles,exact_validated_executions=counts,timing_windows=windows,fp64_worst_ratio=worst,clean_groups=12,full_model_gain_established=False)
Path(sys.argv[2]).write_bytes((json.dumps(summary,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in summary.items() if k!='profiles'}))
