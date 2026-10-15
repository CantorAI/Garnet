"""Native ABBA for native BF16 body projection; same frozen workloads and full outputs."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';statuses=[];background_checks=[];records=[];profiles=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
raw_snapshot=snapshot;owned_group=None
# Only the process group created by this runner is exempt during monitoring.
def snapshot():
    retained=[];owned=0
    for line in raw_snapshot().splitlines():
        f=line.split()
        if owned_group is not None and f and f[-1].lower().endswith('/xlang3'):
            try:group=os.getpgid(int(f[0]))
            except ProcessLookupError:continue
            if group==owned_group:owned+=1;continue
        retained.append(line)
    if owned>1:raise RuntimeError('Multiple native workers in owned benchmark group')
    return '\n'.join(retained)

def guard(label):
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')):
            raise RuntimeError('Other GPU probe PID '+f[0])
    (output/(label+'-processes.txt')).write_text(snapshot())
builds={key:r/name for key,name in [('control','build-screened-head'),('candidate','build-screened-prefill-reset')]}
metadata=dict(common_frontend='combined threaded decode and exact-coordinate strip resize on both builds',scope=__doc__,library_sha256={key:hashlib.file_digest((p/'libgarnet.dylib').open('rb'),'sha256').hexdigest() for key,p in builds.items()},source_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [Path(__file__),s/'run_matched_garnet.py',s/'run_threaded_striped_frame_process.py',s/'frame_worker_garnet.py']},assets={})
metadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('compare_parallel_vision.py','compare_vision_global_alpha.py','run_screened_head_profiles.py','run_screened_head_session_reference.py')})
metadata['threaded_decoder_sha256']=hashlib.sha256((r/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib').read_bytes()).hexdigest()
assert metadata['threaded_decoder_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
metadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('frame_threaded_striped.py','frame_webp_threaded.py','frame_preprocessing_striped.py','run_webp_thread_frame_process.py','run_matched_frame_process.py','frame_preprocessing.py')})
assert metadata['library_sha256']['control']=='2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23'
check_metadata=json.loads((r/'logs/screened-prefill-reset-checks-20261015/jobs/metadata.json').read_text());assert metadata['library_sha256']['candidate']==check_metadata['candidate_sha256']
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';frontend=str(r/'comparators/vllm/venv/bin/python')
selected_model=sys.argv[sys.argv.index('--model')+1] if '--model' in sys.argv else None
selected_batch=int(sys.argv[sys.argv.index('--batch')+1]) if '--batch' in sys.argv else None
assert selected_model in (None,'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') and selected_batch in (None,1,2)
reuse=Path(sys.argv[sys.argv.index('--reuse-preflight')+1]) if '--reuse-preflight' in sys.argv else None

for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    if selected_model is not None and selected_model!=model:continue
    vl=model=='Qwen3.5-0.8B';baseline=r/'logs'/('equivalent-f32-qwen35-extended-prefill-20261013' if vl else 'equivalent-f32-finer-prefill-20261013')/'jobs'
    folder=r/'models'/model
    metadata['assets'][model]={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [folder/'model.safetensors',folder/'config.json',folder/'tokenizer.json']}
    metadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('compare_screened_head.py','compare_dirty_prefix_reset.py','run_screened_prefill_reset_profiles.py')})
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for batch in ((selected_batch,) if selected_batch else (1,2)):
        label=model+'-b'+str(batch);case=json.loads((baseline/(label+'-case.json')).read_text());case['requests']=3
        if not vl:case['prefill_exact_tail']=model!='Qwen3-0.6B' or batch==2
        casefile=output/(label+'-case.json');casefile.write_text(json.dumps(case,indent=2))
        if vl:expected=json.loads((baseline/(label+'-garnet-0/protocol/command.json.event-0-frame_result.json')).read_text())
        else:expected=next(json.loads(v.split(' ',1)[1]) for v in (baseline/(label+'-garnet-0/output.log')).read_text().splitlines() if v.startswith('matched_result '))
        (output/(label+'-expected.json')).write_text(json.dumps(expected,indent=2))
        for policy,repeat in [('control',0),('candidate',0),('candidate',1),('control',1)]:
            name=label+'-'+policy+'-'+str(repeat);directory=output/name;directory.mkdir();guard(name+'-before')
            if vl:command=[frontend,str(s/'run_threaded_striped_frame_process.py'),runtime,str(builds[policy]),str(r/'source'),str(folder),str(casefile),'garnet',str(directory/'protocol'),'combined']
            else:command=[runtime,str(s/'run_matched_garnet.py'),str(builds[policy]),str(r/'source'),str(folder),str(casefile)]
            print('Starting exclusive native worker',name,flush=True)
            samples=[];begin=time.monotonic()
            with (directory/'output.log').open('w') as stream:
                process=subprocess.Popen(['/usr/bin/time','-l',*command],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);owned_group=process.pid
                try:
                    while process.poll() is None:
                        guard(name+'-monitor-current');samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                        if time.monotonic()-begin>900:raise TimeoutError('Owned native worker exceeded deadline')
                        time.sleep(.5)
                    code=process.returncode
                except Exception as error:
                    status=dict(name=name,command=['/usr/bin/time','-l',*command],exit_code=1,interruption=str(error));statuses.append(status)
                    (directory/'exit.json').write_text(json.dumps(status,indent=2));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
                    if process.poll() is None:
                        assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
                finally:
                    owned_group=None;(directory/'monitor.json').write_text(json.dumps(samples,indent=2))
            status=dict(name=name,command=['/usr/bin/time','-l',*command],exit_code=code);statuses.append(status)
            (directory/'exit.json').write_text(json.dumps(status,indent=2));(output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert code==0,status
            guard(name+'-after')
            import re
            rss=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',(directory/'output.log').read_text(),re.M)
            assert len(rss)==1,'Missing or ambiguous time-l RSS record'
            peak_rss=int(rss[0])
            prefix='matched_frame_result ' if vl else 'matched_result '
            values=[json.loads(v.split(' ',1)[1]) for v in (directory/'output.log').read_text().splitlines() if v.startswith(prefix)];assert len(values)==3
            for value in values:
                assert value['prompt_token_ids']==expected['prompt_token_ids'] and value['token_ids']==expected['token_ids'] and value['finished']==expected['finished']
                assert value['case']==case
                if vl:assert value['vision_reencoded'] and value['patch_sha256']==case['image_metadata']['patch_sha256']
                if vl:
                    assert value['frontend_policy']=='combined' and value['frontend_route']=='threaded-striped'
                    assert not value['parallel_vision_context_enabled']
                    assert value['vision_parallel_dispatches']==0
                    assert not value['body_shape_enabled'] and not value['fused_head_argmax_enabled']
                    previous=[v for v in records if v['model_name']==model and v['case']['batch']==batch]
                    assert all(v['rgb_sha256']==value['rgb_sha256'] for v in previous)

                assert value['attention_prefix_enabled']
                assert value['global_half_alpha_enabled']
                assert value['screened_head_enabled']
                counts=value['screened_head_counts'];assert len(counts)==value['decode_steps_after_first']+1
                assert counts==[1]*len(counts)
                assert not value['fused_head_argmax_enabled'] and not value['body_shape_enabled'] and not value['deferred_state_zero_enabled'] and not value['reused_zero_source_enabled']

                assert not value['vision_global_alpha_enabled']
                config=json.loads((folder/'config.json').read_text());config=config.get('text_config',config)
                assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0
                assert value['prefill_stream_enabled']==(policy=='candidate') and value['dirty_prefix_reset_enabled']==(policy=='candidate')
                assert value['decode_partial_submissions']==0
                submissions=((config['num_hidden_layers']-1)//4)*sum(chunk>1 for chunk in value['prefill_chunk_plan']) if policy=='candidate' else 0
                assert value['prefill_profile'].get('prefill_partial_submissions',0)==submissions
                assert value['vision_global_alpha_dispatches']==0
                if policy=='candidate':
                    kinds=config.get('layer_types',['full_attention']*config['num_hidden_layers'])
                    full=kinds.count('full_attention');linear=kinds.count('linear_attention')
                    assert value['dirty_prefix_reset_states']==2*full
                    assert value['dirty_prefix_reset_fallback_states']==2*linear
                    extent=0 if value['request']==0 else len(value['prompt_token_ids'])+value['decode_steps_after_first']
                    dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
                    columns=2*full*config['num_key_value_heads']*extent*dim
                    if linear:
                        channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
                        columns+=linear*(channels*config['linear_conv_kernel_dim']+config['linear_num_value_heads']*config['linear_key_head_dim']*config['linear_value_head_dim'])
                    assert value['dirty_prefix_reset_bytes']==batch*columns*4
                else:assert value['dirty_prefix_reset_bytes']==value['dirty_prefix_reset_states']==value['dirty_prefix_reset_fallback_states']==0
                assert not value['single_coder_head_enabled'] and not value['joined_global_mlp_enabled'] and not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']
                value.update(model_name=model,build_policy=policy,repeat=repeat,peak_individual_rss_bytes=peak_rss);records.append(value)
            print('Completed exact native worker',name,flush=True)
        warm={policy:[v for v in records if v['model_name']==model and v['case']['batch']==batch and v['build_policy']==policy and v['request']>0] for policy in builds}
        means={policy:dict(requests=len(values),ttft_ms=statistics.mean(v['ttft_seconds'] for v in values)*1000,decode_tps=statistics.mean(v['decode_wall_steps_per_second'] for v in values),initialization_s=statistics.mean(v['initialization_seconds'] for v in values),peak_individual_rss_bytes=statistics.mean(v['peak_individual_rss_bytes'] for v in values)) for policy,values in warm.items()}
        assert all(x['requests']==4 for x in means.values())
        profiles.append(dict(model=model,batch=batch,means=means,decode_ratio=means['candidate']['decode_tps']/means['control']['decode_tps'],ttft_fraction=means['candidate']['ttft_ms']/means['control']['ttft_ms'],exact_complete_outputs=True))
        (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],profiles=profiles,records=records),indent=2));print(json.dumps(profiles[-1]),flush=True)
expected_profiles=(1 if selected_model else 3)*(1 if selected_batch else 2)
assert len(statuses)==expected_profiles*4 and len(records)==expected_profiles*12 and len(profiles)==expected_profiles
