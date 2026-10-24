from pathlib import Path
import hashlib,json,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
failed=r/'results/nvfp4-down-full-qualification';wf=json.loads((failed/'workflow.json').read_text(encoding='utf8'));assert wf['status']=='failed' and wf['stages'][0]['status']=='passed' and wf['stages'][1]['exit_code']==1
out=r/'results/nvfp4-down-state-controls';assert not out.exists();digest=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();assert digest==wf['library_sha256']
for mode in ('original','joined'):
 batch=r/('test-nvfp4-down-control-'+mode+'.bat');cmd=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(r/('results/nvfp4-down-control-'+mode+'-resources')),'--','cmd.exe','/d','/c',str(batch)]
 code=run(cmd,r/('results/nvfp4-down-control-'+mode+'-ledger'),'correctness','candidate-off-'+mode,[batch,r/'build/bin/garnet.dll'])
 if code:raise SystemExit(code)
 report=json.loads((out/(mode+'-report.json')).read_text(encoding='utf8'));assert report['status']=='passed' and report['library_sha256']==digest
old=json.loads((failed/'state/original-report.json').read_text(encoding='utf8'));active=json.loads((failed/'state/joined-report.json').read_text(encoding='utf8'))
original=json.loads((out/'original-report.json').read_text(encoding='utf8'));joined=json.loads((out/'joined-report.json').read_text(encoding='utf8'))
result={'status':'controls_completed','library_sha256':digest,'candidate_flag':0,'all_inputs_equal':old['inputs']==active['inputs']==original['inputs']==joined['inputs'],'default_original_equal_old_teacher':original['cases']==old['cases'],'default_joined_equal_original':joined['cases']==original['cases'],'active_candidate_equal_default_joined':active['cases']==joined['cases'],'scope':'Untimed same-binary option-off controls after failed option-on full-state gate. Original failure preserved; no quality/tolerance/performance waiver.'}
(out/'control-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
