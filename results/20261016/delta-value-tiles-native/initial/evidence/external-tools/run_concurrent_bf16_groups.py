"""Exclusive grouped native projection experiment; no inference speed claim."""
from pathlib import Path
import statistics
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
native=root/'source/src/backends/metal/metal_context.mm'
paths=[native,s/'concurrent_bf16_group_gpu.mm',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},
    policies={'0':'Separate serial encoders','1':'One serial encoder','2':'One concurrent encoder, independent writes'},
    numerical_bounds='Every candidate complete output and padding bit-exact versus original shader; original FP64 gamma(ceil(K/32)+32)*sumabs+1e-7',
    hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},
    versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
text=native.read_text();shader=out/'native-decode.metal'
shader.write_bytes(('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+text[text.index('kernel void bf16_decode('):text.index('kernel void bf16_gemm(')]).encode())
metadata['native_shader_sha256']=digest(shader)
binary=out/'concurrent_bf16_group_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(s/'concurrent_bf16_group_gpu.mm'),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);profiles=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    old=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');info=json.loads((old/'manifest.json').read_text())
    checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256']
    def select(suffix):
        matches=[c for c in info['cases'] if c['tensor'].endswith(suffix)];assert len(matches)==1;return matches[0]
    groups=[dict(name='mlp-gate-up',cases=[select('.mlp.gate_proj.weight'),select('.mlp.up_proj.weight')])]
    if model!='Qwen3.5-0.8B':groups.append(dict(name='attention-qkv',cases=[select('.self_attn.'+p+'_proj.weight') for p in ('q','k','v')]))
    tail=next(c for c in info['cases'] if c['tensor'].startswith('synthetic') and c['inner']==37)
    groups.append(dict(name='irregular-readonly-pair',cases=[tail,tail]))
    chosen={c['name']:c for g in groups for c in g['cases']};assets=out/(model+'-assets');assets.mkdir()
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);start=8+length
        for name,c in chosen.items():
            assert digest(old/(name+'.bin'))==c['sha256']
            if not c['tensor'].startswith('synthetic'):
                e=header[c['tensor']];a,b=e['data_offsets'];assert e['dtype']=='BF16' and e['shape']==[c['columns'],c['inner']] and hashlib.sha256(mapping[start+a:start+b]).hexdigest()==c['sha256']
            (assets/(name+'.bin')).write_bytes((old/(name+'.bin')).read_bytes())
    grouped=dict(groups=groups,checkpoint_sha256=info['checkpoint_sha256'])
    (assets/'groups.json').write_bytes((json.dumps(grouped,indent=2)+'\n').encode());metadata['assets'][model]=grouped
    (out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(shader),str(assets)])
    records=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[v for v in records if v['kind']=='check'];assert len(checks)==2*len(groups)
    for c in checks:
        assert c['bits_exact'] and c['padding_preserved'] and c['input_variants']==4 and c['fp64_worst_ratio']<=1 and c['validated_executions']>=252
        for mode in (1,2):
            windows=[v for v in records if v['kind']=='timing' and v['group']==c['group'] and v['rows']==c['rows'] and v['candidate_mode']==mode];assert len(windows)==8
            for rnd in (0,1):assert [v['mode'] for v in sorted([v for v in windows if v['round']==rnd],key=lambda v:v['window'])]==[0,mode,mode,0]
            assert all(len(v[k])==12 for v in windows for k in ('encode_ms','wall_ms','gpu_ms'))
            means={policy:{k:statistics.mean(x for v in windows if (v['mode']==mode)==candidate for x in v[k]) for k in ('encode_ms','wall_ms','gpu_ms')} for policy,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,group=c['group'],batch=c['rows'],candidate_mode=mode,checks=c,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    print('Completed independent projection groups',model,flush=True)
    (out/'summary.json').write_bytes((json.dumps(dict(scope=__doc__,profiles=profiles,full_goal_met=False),indent=2)+'\n').encode())
