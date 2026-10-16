"""Complete learned projection/norm inputs before fixed decode/prefill paired-RoPE timings."""
from pathlib import Path
import re,shlex
from make_paired_norm_rope import context,header
p=Path(__file__).with_name('run_constant_bf16_probe.py');text=p.read_text();ns={'__file__':__file__}
exec(compile(text[:text.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
selected=root/'vector-vision-native-investigation-20261016';base=selected/'source/src/backends/metal';build=selected/'build-vector-vision-width4';assert digest(build/'libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
primitive=root/'validation-audits/paired-norm-rope-primitive-20261016';audit=json.loads((primitive/'verification.json').read_text());assert audit['synthetic_changed_input_cases']==60 and audit['original_fp32_output_bits_checked']==3710720
prior=json.loads((primitive/'evidence/logs/paired-norm-rope-primitive-20261016/jobs/metadata.json').read_text());assert [v for k,v in prior['source_sha256'].items() if k.endswith('/make_paired_norm_rope.py')]==[digest(s/'make_paired_norm_rope.py')]
cpu=root/'validation-audits/qkv-union-half-admission-20261016/evidence/logs/qkv-union-half-admission-20261016/report.json';report=json.loads(cpu.read_text());assets={}
for model in report['models']:
 checkpoint=Path(model['checkpoint']);assert digest(checkpoint)==model['checkpoint_sha256'];config=root/'models'/model['model']/'config.json';raw_config=json.loads(config.read_text());c=raw_config.get('text_config',raw_config);vl=model['model']=='Qwen3.5-0.8B';coder=model['model']=='Qwen2.5-Coder-0.5B-Instruct'
 dim=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);rope=c.get('rope_parameters',{});rotary=int(dim*rope.get('partial_rotary_factor',1.));sections=rope.get('mrope_section',[rotary//2,0,0])
 model['rope']=dict(query_heads=c['num_attention_heads'],key_heads=c['num_key_value_heads'],dim=dim,rotary=rotary,normalize=not coder,weight_offset=1 if vl else 0,theta=rope.get('rope_theta',c.get('rope_theta',1000000.)),sections=sections,epsilon=c['rms_norm_eps'],query_gated=vl,prefill_rows=[80,160] if vl else ([26,52] if coder else [24,44]))
 with checkpoint.open('rb') as stream:
  length=struct.unpack('<Q',stream.read(8))[0];header_info=json.loads(stream.read(length));data_start=8+length
  for group in model['groups']:
   group['norms']=[];stem=group['sides'][0]['tensor'].rsplit('.q_proj.weight',1)[0]
   for part in ('q','k'):
    name=stem+'.'+part+'_norm.weight'
    if coder:assert name not in header_info;group['norms'].append(dict(tensor=None,ones=True));continue
    item=header_info[name];assert item['dtype'] in ('BF16','F32') and item['shape']==[dim];lo,hi=item['data_offsets'];stream.seek(data_start+lo);raw=stream.read(hi-lo);assert len(raw)==hi-lo
    group['norms'].append(dict(tensor=name,ones=False,dtype=item['dtype'],data_offset=data_start+lo,bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest()))
 assets[model['model']]=dict(checkpoint_sha256=model['checkpoint_sha256'],config_sha256=digest(config),config=raw_config)
report_path=out/'learned-report.json';report_path.write_text(json.dumps(report,indent=2)+'\n');assert sum(len(m['groups']) for m in report['models'])==58
generated=out/'generated';generated.mkdir();(generated/'metal_context.h').write_bytes(header((base/'metal_context.h').read_text()).encode());(generated/'metal_context.mm').write_bytes(context((base/'metal_context.mm').read_text()).encode())
flags_path=build/'CMakeFiles/garnet_metal.dir/flags.make';flags=flags_path.read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')];assert defines==prior['defines']
paths=[Path(__file__),s/'make_paired_norm_rope.py',s/'paired_norm_rope_learned.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',s/'launch_paired_norm_rope_learned.py',s/'launch_quiet_prefill_residual_norm_probe.py',flags_path,*[p for p in base.iterdir() if p.suffix in ('.h','.mm')]]
meta=dict(source_sha256={p.as_posix():digest(p) for p in paths},generated_sha256={p.name:digest(p) for p in generated.iterdir()},defines=defines,reference_library_sha256=digest(build/'libgarnet.dylib'),learned_report_sha256=digest(report_path),assets=assets,timing_design=dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,null_before_comparison=True,all_samples_retained=True,timed_layers={'Qwen3-0.6B':[0,27],'Qwen2.5-Coder-0.5B-Instruct':[0,23],'Qwen3.5-0.8B':[3,23]},endpoint_selection='First/last separate full-attention layer; all four decode/prefill row profiles, fixed before execution'),scope=__doc__,limitations='Learned native Q/K projections and actual norm weights on changed structured hidden inputs. Host fixtures reused only for kernel-level alternating measurements; no prefix/input cache or full-model inference speed claim. Model init/first/warm behavior remains separate.',full_goal_met=False)
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');binary=out/'paired_norm_rope_learned_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(generated),'-I'+str(base),str(s/'paired_norm_rope_learned.mm'),str(generated/'metal_context.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
meta['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');run('numeric',[str(binary),str(report_path),'numeric'])
records=[json.loads(v) for v in (out/'numeric.log').read_text().splitlines()];assert records[-1]['kind']=='complete' and records[-1]['profiles']==232 and records[-1]['changed_input_cases']==928
run('timing',[str(binary),str(report_path),'timing'])
for name,sha in meta['source_sha256'].items():assert digest(Path(name))==sha
print('Learned paired norm-RoPE complete; independent audit/selection and native validation still required.',flush=True)
