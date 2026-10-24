from pathlib import Path
import hashlib,json,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
from joined_model_gates import validate_joined_model_gates
component=json.loads((r/'results/nvfp4-decode-only-attempt2/workflow-passed.json').read_text(encoding='utf8'));assert component['status']=='component_gates_and_native_build_passed'
dll=r/'build/bin/garnet.dll';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=sha(dll);assert digest==component['library_sha256']
out=r/'results/nvfp4-decode-only-full-qualification';out.mkdir(exist_ok=False)
report={'status':'starting','library_sha256':digest,'source':json.loads((r/'study-state.json').read_text(encoding='utf8')),'requested_policy':'v3 decode-only tag2; baselineT128','stages':[],'scope':'Untimed same-binary full state and retained target/vector goldens; no MTP/HTTP/performance admission.'};path=out/'workflow.json'
def read(name):return json.loads((out/name).read_text(encoding='utf8'))
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
 for name,flag,result,want in [('state-original',0,'state/original-report.json','passed'),('state-candidate',1,'state/joined-report.json','passed'),('baseline',0,'baseline/report.json','smoke_passed'),('candidate',1,'candidate/report.json','smoke_passed')]:
  batch=r/('test-nvfp4-decode-only-'+name+'.bat');assert sha(dll)==digest
  row={'stage':name,'candidate_requested':bool(flag),'batch_sha256':sha(batch),'status':'running'};report['stages'].append(row);save()
  cmd=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/(name+'-resources')),'--','cmd.exe','/d','/c',str(batch)]
  code=run(cmd,out/(name+'-ledger'),'correctness','decode-only-full-'+name,[batch,dll,r/'nvfp4-decode-only-source.zip']);row.update(exit_code=code,status='terminal');save()
  if code:raise RuntimeError('Decode-only full stage failed: '+name)
  trial=read(result);assert trial['status']==want and trial['library_sha256']==digest
  if name=='state-original':
   old=json.loads((r/'results/nvfp4-down-state-controls/original-report.json').read_text(encoding='utf8'));assert old['inputs']==trial['inputs'] and old['cases']==trial['cases'],'Default full state changed after rebuild'
  row['status']='passed';save()
 golden=json.loads((r/'results/shared-decoder-full-qualification/report.json').read_text(encoding='utf8'))
 report['model_gates']=validate_joined_model_gates(read('state/original-report.json'),read('state/joined-report.json'),read('candidate/report.json'),golden,digest)
 assert sha(dll)==digest;report['status']='same_binary_state_and_target_goldens_passed'
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:save()
