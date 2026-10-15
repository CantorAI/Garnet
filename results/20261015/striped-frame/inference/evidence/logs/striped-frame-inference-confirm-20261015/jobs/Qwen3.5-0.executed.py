"""Exact-coordinate striped resize applied equally to Garnet/vLLM, sequential factorial ABBA.

This additional precision study does not establish default BF16 speedups.
"""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);model=sys.argv[3];batch=int(sys.argv[4])
assert model=='Qwen3.5-0.8B' and batch in (1,2)
output.mkdir(parents=True,exist_ok=False);s=r/'source/test2026/metal';statuses=[];records=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
real_subprocess=subprocess;owned_group=None
def owned(pid):
    if owned_group is None:return False
    try:return os.getpgid(pid)==owned_group
    except ProcessLookupError:return False
class ProcessTools:
    def __getattr__(self,name):return getattr(real_subprocess,name)
    def check_output(self,command,**kwargs):
        result=real_subprocess.check_output(command,**kwargs)
        if owned_group is not None and command==['/bin/ps','-axo','pid=,args=']:
            result='\n'.join(v for v in result.splitlines() if not v.split() or not owned(int(v.split()[0])))
        return result
subprocess=ProcessTools()
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard;raw_snapshot=snapshot
def snapshot():
    kept=[];native_count=0
    for line in raw_snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('/xlang3','/vllm','/ollama')) and owned(int(fields[0])):
            if fields[-1].lower().endswith('/xlang3'):native_count+=1
            continue
        kept.append(line)
    if native_count>1:raise RuntimeError('Multiple owned native inference workers')
    return '\n'.join(kept)
def guard(label):
    import shutil
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('_gpu','_probe')):
            raise RuntimeError('Other GPU probe PID '+fields[0])
    (output/(label+'-processes.txt')).write_text(snapshot())
vl=model=='Qwen3.5-0.8B';native=r/'models'/model;comparator=r/'models-exact-f32'/model;build=r/'build-global-half-alpha'
proof_path=comparator/'exact-value-equivalence.json';proof=json.loads(proof_path.read_text())
assert proof['all_values_bit_exact'] and all(v['all_values_bit_exact'] for v in proof['tensors'])
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert digest(native/'model.safetensors')==proof['source_sha256'] and digest(comparator/'model.safetensors')==proof['destination_sha256']
(output/'exact-value-equivalence.json').write_bytes(proof_path.read_bytes())
baseline=r/'logs'/('equivalent-f32-qwen35-extended-prefill-20261013' if vl else 'equivalent-f32-finer-prefill-20261013')/'jobs';label=model+'-b'+str(batch)
case=json.loads((baseline/(label+'-case.json')).read_text());case['requests']=3;assert case['comparator_dtype']=='float32'
if not vl:case['prefill_exact_tail']=model!='Qwen3-0.6B' or batch==2
if vl:
    expected=json.loads((baseline/(label+'-garnet-0/protocol/command.json.event-0-frame_result.json')).read_text())
    native_prompt=json.loads((baseline/(label+'-garnet-0/protocol/ready.json')).read_text());case['native_prompt']=native_prompt
else:
    text=(baseline/(label+'-garnet-0/output.log')).read_text()
    expected=next(json.loads(v.split(' ',1)[1]) for v in text.splitlines() if v.startswith('matched_result '))
    native_prompt=next(json.loads(v.split(' ',1)[1]) for v in text.splitlines() if v.startswith('matched_prompt '))
casefile=output/'case.json';casefile.write_text(json.dumps(case,indent=2))
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';python=str(r/'comparators/vllm/venv/bin/python')
metadata=dict(scope=__doc__,counts_as_default_bf16_speedup=False,native_library_sha256=digest(build/'libgarnet.dylib'),learned_values_bit_exact=True,
    source_checkpoint_sha256=proof['source_sha256'],comparator_checkpoint_sha256=proof['destination_sha256'],equivalence_proof_sha256=digest(proof_path),
    source_sha256={p.name:digest(p) for p in [Path(__file__),s/'run_matched_garnet.py',s/'run_matched_vllm.py',s/'frame_worker_garnet.py',s/'frame_worker_vllm.py',s/'run_striped_frame_process.py',s/'benchmark_matched_process.py',s/'frame_preprocessing.py',s/'frame_preprocessing_control.py']})
assert metadata['native_library_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
metadata['source_sha256'].update({n:digest(s/n) for n in ('frame_preprocessing_striped.py','compare_frame_frontend_inference.py','run_matched_frame_process.py')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
native_log=None
for backend,frontend_policy,repeat in [('garnet','selected',0),('vllm-metal','selected',0),('vllm-metal','striped',0),('garnet','striped',0),('garnet','striped',1),('vllm-metal','striped',1),('vllm-metal','selected',1),('garnet','selected',1)]:
    name=backend+'-'+frontend_policy+'-'+str(repeat);folder=output/name;folder.mkdir();guard(name+'-before')
    if vl:command=[python,str(s/'run_striped_frame_process.py'),runtime if backend=='garnet' else python,str(build),str(r/'source'),str(native if backend=='garnet' else comparator),str(casefile),backend,str(folder/'protocol'),frontend_policy]
    elif backend=='garnet':command=[runtime,str(s/'run_matched_garnet.py'),str(build),str(r/'source'),str(native),str(casefile)]
    else:command=[python,str(s/'run_matched_vllm.py'),str(comparator),str(casefile),str(native_log)]
    command=[sys.executable,str(s/'benchmark_matched_process.py'),*command];samples=[];begin=time.monotonic()
    print('Starting sequential runtime',label,name,flush=True)
    with (folder/'output.log').open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);owned_group=process.pid
        try:
            while process.poll() is None:
                guard(name+'-monitor-current');samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                if time.monotonic()-begin>900:raise TimeoutError('Owned inference worker exceeded deadline')
                time.sleep(.5)
            code=process.returncode
        except Exception as error:
            status=dict(name=name,command=command,exit_code=1,interruption=str(error));statuses.append(status)
            (folder/'exit.json').write_text(json.dumps(status,indent=2));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
            if process.poll() is None:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:owned_group=None;(folder/'monitor.json').write_text(json.dumps(samples,indent=2))
    status=dict(name=name,command=command,exit_code=code);statuses.append(status)
    (folder/'exit.json').write_text(json.dumps(status,indent=2));(output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert code==0,status
    guard(name+'-after');text=(folder/'output.log').read_text();assert 'Segfault encountered' not in text
    prefix='matched_frame_result ' if vl else 'matched_result '
    values=[json.loads(v.split(' ',1)[1]) for v in text.splitlines() if v.startswith(prefix)];assert len(values)==3
    processes=[json.loads(v.split(' ',1)[1]) for v in text.splitlines() if v.startswith('matched_process ')];assert processes and processes[-1]['exit_code']==0
    for value in values:
        ids=[list(v) for v in value['token_ids']]
        if backend=='vllm-metal':
            assert not value['prefix_caching'] and not value['speculative_decoding'] and value['configured_model_dtype'] in ('torch.float32','float32')
            if not vl:
                eos=json.loads((comparator/'config.json').read_text())['eos_token_id'];stops=native_prompt['stop_token_ids'];assert eos in stops
                for row,done,reason in zip(ids,value['finished'],value['stop_reasons']):
                    if done and (not row or row[-1] not in stops):
                        stop=reason if isinstance(reason,int) else eos if reason is None else None
                        assert stop in stops,'Unverified terminal stop reason';row.append(stop)
        assert ids==expected['token_ids'] and value['prompt_token_ids']==expected['prompt_token_ids'] and value['finished']==expected['finished']
        assert value['decode_steps_after_first']==max(map(len,ids))-1
        assert value['case']==case and value['frontend_policy']==frontend_policy
        assert value['frontend_route']==('parallel' if frontend_policy=='striped' else 'selected') and value['frontend_setup_seconds']>=0
        if backend=='garnet':
            expected_plan=([24] if batch==1 else [22]) if model=='Qwen3-0.6B' else [26] if model=='Qwen2.5-Coder-0.5B-Instruct' else [80] if batch==1 else [32,32,16]
            assert value['prefill_chunk_plan']==expected_plan
            assert value['global_half_alpha_enabled'] and not value['body_shape_enabled']
            assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0
        if vl:
            assert value['patch_sha256']==case['image_metadata']['patch_sha256']
            if backend=='garnet':assert value['vision_reencoded']
            else:assert len(set(value['encoder_cache_identities']))==batch
        value.update(normalized_token_ids=ids,model_name=model,backend=backend,repeat=repeat,process=processes[-1]);records.append(value)
    if backend=='vllm-metal':assert len({identity for value in values for identity in value['encoder_cache_identities']})==batch*3,'Encoder identities reused across requests'
    if backend=='garnet':native_log=folder/'output.log'
    print('Completed exact runtime',label,name,flush=True)
policies={}
for policy in ('selected','striped'):
    warm={backend:[v for v in records if v['backend']==backend and v['frontend_policy']==policy and v['request']>0] for backend in ('garnet','vllm-metal')}
    means={backend:dict(requests=len(values),ttft_ms=statistics.mean(v['ttft_seconds'] for v in values)*1000,decode_tps=statistics.mean(v['decode_wall_steps_per_second'] for v in values),preprocessing_ms=statistics.mean(v['common_preprocessing_seconds'] for v in values)*1000,initialization_s=statistics.mean(v['initialization_seconds'] for v in values),peak_child_rss_bytes=statistics.mean(v['process']['peak_rss_bytes'] for v in values)) for backend,values in warm.items()}
    assert all(v['requests']==4 for v in means.values())
    ratio=means['garnet']['decode_tps']/means['vllm-metal']['decode_tps'];fraction=means['garnet']['ttft_ms']/means['vllm-metal']['ttft_ms']
    policies[policy]=dict(means=means,decode_ratio=ratio,ttft_fraction=fraction,decode_target_met=ratio>=2,ttft_target_met=fraction<=.5)
assert len(records)==24
frontend_effects={backend:dict(ttft_fraction=policies['striped']['means'][backend]['ttft_ms']/policies['selected']['means'][backend]['ttft_ms'],decode_ratio=policies['striped']['means'][backend]['decode_tps']/policies['selected']['means'][backend]['decode_tps']) for backend in ('garnet','vllm-metal')}
summary=dict(model=model,batch=batch,policies=policies,frontend_effects=frontend_effects,exact_complete_outputs=True,counts_as_default_bf16_speedup=False,records=records,scope=__doc__)
(output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
