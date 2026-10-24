from pathlib import Path
import hashlib,json,shutil,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
component=r/'results/nvfp4-down-runtime-attempt2';done=json.loads((component/'workflow-passed.json').read_text(encoding='utf8'));assert done['status']=='component_gates_and_native_build_passed'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();dll=r/'build/bin/garnet.dll';digest=sha(dll);assert digest==done['library_sha256']
out=r/'results/nvfp4-down-full-qualification';out.mkdir(exist_ok=False);state=out/'state';state.mkdir()
for name in ('original-report.json','text-input.f32','image-input.f32'):shutil.copy2(r/'results/joined-program-equivalence'/name,state/name)
report={'status':'starting','library_sha256':digest,'source':json.loads((r/'study-state.json').read_text(encoding='utf8')),'stages':[],'scope':'Untimed retained full-state and target-token/vector goldens; no MTP, HTTP, broad quality or speed admission.'};path=out/'workflow.json'
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
 for name,flag,result,want in [('baseline',0,'baseline/report.json','smoke_passed'),('state',1,'state/joined-report.json','passed'),('candidate',1,'candidate/report.json','smoke_passed')]:
  batch=r/('test-nvfp4-down-full-'+name+'.bat');assert sha(dll)==digest
  row={'stage':name,'requested_candidate_policy':bool(flag),'batch_sha256':sha(batch),'status':'running'};report['stages'].append(row);save()
  cmd=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/(name+'-resources')),'--','cmd.exe','/d','/c',str(batch)]
  code=run(cmd,out/(name+'-ledger'),'correctness','full-down-'+name,[batch,dll,r/'nvfp4-down-runtime-source.zip',r/'nvfp4-down-runtime-fix-source.zip'])
  row['exit_code']=code;row['status']='terminal';save()
  if code:raise RuntimeError('Full-model stage failed: '+name)
  result_report=json.loads((out/result).read_text(encoding='utf8'));assert result_report['status']==want and result_report['library_sha256']==digest
  row['status']='passed';save()
 assert sha(dll)==digest;report['status']='full_state_and_target_goldens_passed'
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:save()
