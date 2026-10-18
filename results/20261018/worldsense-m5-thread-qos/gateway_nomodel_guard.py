"""Actual native parent gateway lifecycle/error guard; no model inference."""
import sys,os,time,json,signal
from pathlib import Path
mode=sys.argv[1];w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-thread-qos');source=Path('/Users/shawn/CantorAI/out/research/worldsense-thread-qos/source');build=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-thread-qos-'+mode);sys.path.insert(0,str(source));sys.path.insert(0,str(build))
import garnet
from qos_call_guard import qos_valid
assert 'native_text_worker'in garnet.backend_info();root=w/'invalid-text-fixture';root.mkdir(exist_ok=True);(root/'config.json').write_text(json.dumps(dict(model_type='qwen3',hidden_size=2048,num_hidden_layers=28)))
def info():return json.loads(garnet.native_text_worker_info_json(''))
def serve():return json.loads(garnet.serve_model(str(root),str(source/'xModel/qwen/metal_dense'),str(w/'invalid-text-cache'),json.dumps(dict(maxInputTokens=64,maxOutputTokens=8,textPrefixCache=False)),'bad-text'))
records=[]
try:
 assert info()['children']==[]
 status=serve();assert not status['ready']and status.get('error_message');first=info();assert len(first['children'])==1;child=first['children'][0];pid=child['support']['pid'];birth=child['birth_id'];assert child['support']['parent_pid']==os.getpid()and child['models']==[]and child['infer_calls']==0
 records.append(dict(case='spawn_identity_and_failed_model_load',status=status,info=first))
 response=json.loads(garnet.infer_json('test','',2,'bad-text'));assert response['status']=='error'and response['native_worker_call_index']==1;qos_valid(response)
 after=info();assert after['children'][0]['infer_calls']==1 and after['children'][0]['birth_id']==birth;records.append(dict(case='single_failed_infer_no_retry',response=response,info=after))
 rejected=False
 try:garnet.native_text_worker_state_json('bad-text')
 except Exception as error:rejected='model not ready'in str(error)
 assert rejected and info()['children'][0]['birth_id']==birth
 assert json.loads(garnet.native_text_worker_close_json('bad-text'))['closed'];assert not serve()['ready']
 policy=json.loads(garnet.set_benchmark_prefill_policy(False));assert not policy['supported'];records.append(dict(case='state_error_recovery_close_serve_policy',info=info()))
 # The PID is an owned child validated by the manager/nonce above. No unrelated jobs.
 current=info()['children'][0];assert current['support']['pid']==pid and current['birth_id']==birth and current['support']['parent_pid']==os.getpid();os.kill(pid,signal.SIGTERM);time.sleep(.1)
 rejected=False
 try:garnet.infer_json('test','',2,'bad-text')
 except Exception as error:rejected='explicit reload required'in str(error)
 assert rejected and info()['children']==[]
 assert not serve()['ready'];new=info()['children'][0];assert new['birth_id']!=birth and new['infer_calls']==0;records.append(dict(case='owned_peer_exit_no_hidden_retry_explicit_restart',info=info()))
 assert garnet.shutdown();assert info()['children']==[];records.append(dict(case='bounded_explicit_shutdown',info=info()))
 (w/('gateway-nomodel-'+mode+'.json')).write_text(json.dumps(dict(all_cases_pass=True,records=records,scope='Native parent/RAW worker plumbing and failed setup only. No loaded model/output/state/performance proof.'),indent=2)+'\n');print('Gateway native ownership/error/no-retry/explicit-restart/shutdown PASS '+mode,flush=True)
finally:garnet.shutdown()
