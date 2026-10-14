"""Exclusive component gate; no model serving or vLLM gain claim."""
from pathlib import Path
import sys
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text()
ns={'__file__':__file__};exec(compile(base[:base.index("native=root/")],__file__,'exec'),ns);globals().update(ns)
__doc__='Joined lossless input components versus native BF16 decode; component only, no serving or vLLM claim'
native=root/'source/src/backends/metal/metal_context.mm'
paths=[native,s/'joined_activation_decode_gpu.mm',s/'joined_activation_decode.metal',root/'source/src/backends/metal/exact_half_codec.h',Path(__file__),s/'compare_vllm_metal.py',s/'run_constant_bf16_probe.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},
 numerical_bounds='Original 5e-6+5e-5*abs(control) and gamma(ceil(K/32)+32)*sumabs+1e-7 versus independent FP64; no widening',
 control_compile='Original native BF16 shaders with default Metal options',candidate_compile='Fast math disabled for lossless activation split/recombine; MPS FP32 result',
 hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},
 versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
text=native.read_text();shader=out/'native-decode.metal'
shader.write_bytes(('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+text[text.index('kernel void bf16_decode('):text.index('kernel void bf16_gemm(')]).encode())
metadata['native_shader_sha256']=digest(shader);(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
binary=out/'joined_activation_decode_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'joined_activation_decode_gpu.mm'),'-o',str(binary)])
metadata['binary_sha256']=digest(binary)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256']
 for case in info['cases']:assert digest(assets/(case['name']+'.bin'))==case['sha256']
 with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  header_bytes=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+header_bytes]);start=8+header_bytes
  for case in info['cases']:
   if case['tensor'].startswith('synthetic'):continue
   entry=header[case['tensor']];a,b=entry['data_offsets'];assert entry['dtype']=='BF16' and entry['shape']==[case['columns'],case['inner']] and hashlib.sha256(mapping[start+a:start+b]).hexdigest()==case['sha256']
 metadata['assets'][model]=info;(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
 run(model+'-admission',[str(binary),str(shader),str(s/'joined_activation_decode.metal'),str(assets),'reject'])
 run(model+'-gpu',[str(binary),str(shader),str(s/'joined_activation_decode.metal'),str(assets)])
admissions=[];profiles=[];exclusions=[]
for model,info in metadata['assets'].items():
 admission=[json.loads(v) for v in (out/(model+'-admission.log')).read_text().splitlines() if v.startswith('{')]
 assert any(v['kind']=='admission' and v['unsafe_input_rejected'] for v in admission);admissions.append(model)
 rows=[json.loads(v) for v in (out/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')];excluded=[v for v in rows if v['kind']=='excluded'];checks=[v for v in rows if v['kind']=='check']
 assert len(checks)==2*(len(info['cases'])-len(excluded));exclusions.extend(dict(model=model,**v) for v in excluded)
 for checkrow in checks:
  name=checkrow['matrix'];batch=checkrow['rows'];assert checkrow['weight_bits_exact'] and checkrow['activation_bits_exact'] and checkrow['output_padding_preserved'] and checkrow['validated_executions']>=130
  assert all(checkrow[k]<=1 for k in ('comparison_worst_ratio','control_fp64_worst_ratio','candidate_fp64_worst_ratio'))
  warm=next(v for v in rows if v['kind']=='warmup' and v['matrix']==name and v['rows']==batch);assert warm['elapsed_ms']>=250
  windows=[v for v in rows if v['kind']=='timing' and v['matrix']==name and v['rows']==batch];assert len(windows)==8
  for round in (0,1):
   group=sorted((v for v in windows if v['round']==round),key=lambda v:v['window']);assert [v['candidate'] for v in group]==[False,True,True,False]
  assert all(len(v[k])==12 for v in windows for k in ('encode_ms','wall_ms','gpu_ms'))
  means={policy:{metric:sum(x for v in windows if v['candidate']==candidate for x in v[metric])/48 for metric in ('encode_ms','wall_ms','gpu_ms')} for policy,candidate in (('control',False),('candidate',True))}
  profiles.append(dict(model=model,matrix=name,batch=batch,checks=checkrow,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
result=dict(scope=__doc__,wide_exponent_rejection_models=admissions,profiles=profiles,explicit_weight_exclusions=exclusions,full_goal_met=False)
(out/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print('Joined activation component gates completed; no serving speed result',flush=True)
