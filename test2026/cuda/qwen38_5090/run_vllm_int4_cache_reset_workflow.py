from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run;from reference_mtp_equivalence import compare_pair
out=r/'results/vllm-int4-cache-reset-workflow';out.mkdir(exist_ok=False);report={'status':'running','scope':'Fresh ordinary INT4 quality repeat, exact prior same-engine IDs, explicit idle dual-cache reset, matched8-cohort pilot with post-run zero-new-hit gate. No broad/cross-engine or2x claim.'}
path=out/'report.json'
def save():path.write_text(json.dumps(report,indent=2))
save()
try:
 quality=r/'results/vllm-int4-cache-reset-quality';pilot=r/'results/vllm-int4-cache-reset-http-pilot';launcher=p/'serve_vllm_int4_cache_reset.sh';log=r/'results/vllm-int4-cache-reset-server.log'
 code=run([sys.executable,str(p/'run_reference_greedy_candidate.py'),'--study',str(r),'--engine','vllm','--model','qwen38-27b-int4','--url','http://127.0.0.1:18080','--output',str(quality),'--launcher',str(launcher),'--server-log',str(log)],r/'results/vllm-int4-cache-reset-quality-ledger','correctness','cache-reset-fresh-quality',[launcher,p/'run_reference_greedy_candidate.py'])
 if code:raise RuntimeError('Fresh quality capture failed: '+str(code))
 prior=r/'results/vllm-int4-quality/greedy/equivalence.json';fresh=quality/'greedy/equivalence.json';old,new=[json.loads(x.read_text()) for x in (prior,fresh)]
 assert old['status']==new['status']=='passed' and len(old['cases'])==len(new['cases'])==16
 assert all(compare_pair(a,b) for a,b in zip(old['cases'],new['cases'])),'Ordinary full-ID repeat differs'
 report.update(status='quality_passed',prior_greedy_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),fresh_greedy_sha256=hashlib.sha256(fresh.read_bytes()).hexdigest(),exact_prior_same_engine_sequences=16);save()
 code=run([sys.executable,str(p/'run_int4_reference_pilot.py'),str(r),'--engine','vllm','--quality',str(quality),'--output',str(pilot),'--launcher',str(launcher),'--server-log',str(log),'--clear-vllm-mm-caches'],r/'results/vllm-int4-cache-reset-http-ledger','correctness','cache-reset-matched-pilot',[launcher,p/'run_int4_reference_pilot.py',p/'vllm_cache_control.py'])
 if code:raise RuntimeError('Cache-reset pilot failed: '+str(code))
 assert json.loads((pilot/'completed.json').read_text())['cache_counter_delta']['hit_delta']==0
 report['status']='completed';print('VLLM_INT4_CACHE_RESET_WORKFLOW_COMPLETE',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:save()
