from pathlib import Path
import hashlib,json,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run;from reference_mtp_equivalence import compare_pair
out=r/'results/vllm-int4-cache-reset-self-qualified-workflow';out.mkdir(exist_ok=False);path=out/'report.json'
prior=json.loads((r/'results/vllm-int4-cache-reset-prior-repeat-diagnostic.json').read_text(encoding='utf8'));assert prior['matching_prompts']==16 and prior['matching_full_sequences']==15
fresh=r/'results/vllm-int4-cache-reset-quality/greedy/equivalence.json';repeat=r/'results/vllm-int4-cache-reset-self-repeat/equivalence.json';a,b=[json.loads(x.read_text(encoding='utf8')) for x in (fresh,repeat)]
assert a['status']==b['status']=='passed' and len(a['cases'])==len(b['cases'])==16 and all(compare_pair(x,y) for x,y in zip(a['cases'],b['cases']))
assert json.loads((r/'results/vllm-int4-cache-reset-workflow/report.json').read_text(encoding='utf8'))['status']=='failed'
report={'status':'same_process_gates_passed','same_process_full_ids':16,'separate_process_full_ids':15,'qualification_basis':'Current live ordinary engine exact repeat plus4 VLM semantics; old cross-run gate remains failed','files':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (fresh,repeat)},'two_x_eligible':False,'scope':'Separately labelled exploratory timing candidate. Unexplained separate-process output drift retained; no broad/cross-engine or2x claim.'};path.write_text(json.dumps(report,indent=2))
try:
 code=run([sys.executable,str(p/'run_int4_reference_pilot.py'),str(r),'--engine','vllm','--quality',str(r/'results/vllm-int4-cache-reset-quality'),'--output',str(r/'results/vllm-int4-cache-reset-self-qualified-pilot'),'--launcher',str(p/'serve_vllm_int4_cache_reset.sh'),'--server-log',str(r/'results/vllm-int4-cache-reset-server.log'),'--clear-vllm-mm-caches'],r/'results/vllm-int4-cache-reset-self-qualified-http-ledger','correctness','self-repeat-qualified-cache-reset-pilot',[p/'vllm_cache_control.py',p/'run_int4_reference_pilot.py',fresh,repeat])
 if code:raise RuntimeError('Self-qualified control/pilot failed: '+str(code))
 report['status']='completed'
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:path.write_text(json.dumps(report,indent=2))
