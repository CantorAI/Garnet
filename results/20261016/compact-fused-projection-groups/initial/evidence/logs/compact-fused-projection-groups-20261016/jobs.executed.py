'Compact-aware grouped decode plus same-pipeline null; no full-model gain claim.'
from pathlib import Path
from make_compact_fused_projection_groups import generate
import statistics
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
native=root/'source/src/backends/metal/metal_context.mm'
paths=[native,root/'source/src/backends/metal/coalesced_body_block_shader.h',s/'concurrent_bf16_group_gpu.mm',s/'make_compact_fused_projection_groups.py',s/'run_concurrent_bf16_groups.py',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',s/'verify_concurrent_bf16_groups.py',s/'verify_compact_fused_projection_groups.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},
    policies={'0':'Retained compact64/raw128/small64 separate encoders','1':'Compact-aware grouped64, separate outputs','2':'Same retained pipelines and geometry null'},
    numerical_bounds='Every candidate complete output and padding bit-exact versus original shader; original FP64 gamma(ceil(K/32)+32)*sumabs+1e-7',
    hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},
    versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
shader=out/'compact-fused-projections.metal';cpp=out/'compact_fused_projection_group_gpu.mm'
shader_text,cpp_text=generate(native.read_text(),(root/'source/src/backends/metal/coalesced_body_block_shader.h').read_text(),(s/'concurrent_bf16_group_gpu.mm').read_text())
shader.write_bytes(shader_text.encode());cpp.write_bytes(cpp_text.encode())
metadata['native_shader_sha256']=digest(shader);metadata['generated_harness_sha256']=digest(cpp)
metadata['timing_design']=dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,all_samples_retained=True)
binary=out/'compact_fused_projection_group_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(cpp),'-o',str(binary)])
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
    for group in groups:
        for case in group['cases']:
            case['float_key']=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
    grouped=dict(groups=groups,checkpoint_sha256=info['checkpoint_sha256'])
    (assets/'groups.json').write_bytes((json.dumps(grouped,indent=2)+'\n').encode());metadata['assets'][model]=grouped
    (out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(shader),str(assets)])
    records=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[v for v in records if v['kind']=='check'];assert len(checks)==2*len(groups)
    for c in checks:
        assert c['bits_exact'] and c['padding_preserved'] and c['input_variants']==4 and c['fp64_worst_ratio']<=1 and c['validated_executions']>=652
        for mode in (1,2):
            windows=[v for v in records if v['kind']=='timing' and v['group']==c['group'] and v['rows']==c['rows'] and v['candidate_mode']==mode];assert len(windows)==40
            for rnd in range(10):assert [v['mode'] for v in sorted([v for v in windows if v['round']==rnd],key=lambda v:v['window'])]==[0,mode,mode,0]
            assert all(len(v[k])==5 for v in windows for k in ('encode_ms','wall_ms','gpu_ms'))
            means={policy:{k:statistics.mean(x for v in windows if (v['mode']==mode)==candidate for x in v[k]) for k in ('encode_ms','wall_ms','gpu_ms')} for policy,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,group=c['group'],batch=c['rows'],candidate_mode=mode,checks=c,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    print('Completed independent projection groups',model,flush=True)
    (out/'summary.json').write_bytes((json.dumps(dict(scope=__doc__,profiles=profiles,full_goal_met=False),indent=2)+'\n').encode())
