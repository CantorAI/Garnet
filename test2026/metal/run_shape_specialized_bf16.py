"""Exclusive fixed-shape original-math component gate; no inference speed claim."""
from pathlib import Path
import statistics
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
native=root/'source/src/backends/metal/metal_context.mm'
paths=[native,s/'make_shape_specialized_bf16_probe.py',s/'indirect_bf16_projection.mm',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},
    policies='Mode 0 original, mode 1 fixed dimensions/strides, mode 2 also removes tail checks only for complete output tiles',
    numerical_bounds='Every warm/timed output and padding bit-exact; original FP64 gamma(ceil(K/32)+32)*sumabs+1e-7',
    versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
shader=out/'combined.metal';probe=out/'shape_specialized_bf16_gpu.mm';binary=out/'shape_specialized_bf16_gpu'
run('generate',[sys.executable,str(s/'make_shape_specialized_bf16_probe.py'),str(native),str(shader),str(probe)])
metadata['shader_sha256']=digest(shader)
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(probe),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);profiles=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256']
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);start=8+length
        for c in info['cases']:
            assert digest(assets/(c['name']+'.bin'))==c['sha256']
            if not c['tensor'].startswith('synthetic'):
                e=header[c['tensor']];a,b=e['data_offsets'];assert e['dtype']=='BF16' and e['shape']==[c['columns'],c['inner']] and hashlib.sha256(mapping[start+a:start+b]).hexdigest()==c['sha256']
    metadata['assets'][model]=info;(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(shader),str(assets)])
    data=[json.loads(v) for v in (out/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')];checks=[v for v in data if v['kind']=='check'];assert len(checks)==2*len(info['cases'])
    for c in checks:
        assert c['bits_exact'] and c['padding_preserved'] and c['fp64_worst_ratio']<=1 and c['validated_executions']>240
        for pair in ('native','dimensions'):
            windows=[v for v in data if v['kind']=='timing' and v['matrix']==c['matrix'] and v['rows']==c['rows'] and v['pair']==pair];assert len(windows)==8
            for rnd in (0,1):assert [v['policy'] for v in windows if v['round']==rnd]==['control','candidate','candidate','control']
            assert all(len(v[k])==12 for v in windows for k in ('encode_ms','wall_ms','gpu_ms'))
            means={policy:{k:statistics.mean(x for v in windows if v['policy']==policy for x in v[k]) for k in ('encode_ms','wall_ms','gpu_ms')} for policy in ('control','candidate')}
            profiles.append(dict(model=model,matrix=c['matrix'],batch=c['rows'],pair=pair,checks=c,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=__doc__,profiles=profiles,full_goal_met=False),indent=2)+'\n').encode());print('Completed fixed-shape BF16 components',model,flush=True)
