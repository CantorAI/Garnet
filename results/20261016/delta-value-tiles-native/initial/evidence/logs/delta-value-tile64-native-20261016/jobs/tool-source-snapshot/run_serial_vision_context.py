"""Original FP32 vision order: threadgroup and explicit ordered-prefetch variants."""
from pathlib import Path
from make_serial_vision_context import generate
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
import math,statistics
native=root/'source/src/backends/metal/metal_context.mm';paths=[native,Path(__file__),s/'make_serial_vision_context.py',s/'vision_fused_context_gpu.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,limits='Synthetic component only, not full pretrained vision or generated outputs; no inference TPS/TTFT claim',source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},versions=dict(python=sys.version,compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),variants=['original-tg64','original-tg128','original-tg256','ordered-prefetch4-tg128','ordered-prefetch8-tg128'])
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
shader,cpp=generate(native.read_text(),(s/'vision_fused_context_gpu.mm').read_text());shaderfile=out/'serial-vision.metal';cppfile=out/'serial_vision_context_gpu.mm';binary=out/'serial_vision_context_gpu'
shaderfile.write_bytes(shader.encode());cppfile.write_bytes(cpp.encode());metadata.update(shader_sha256=digest(shaderfile),harness_sha256=digest(cppfile))
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(cppfile),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode());profiles=[];allchecks=[]
for variant,label in enumerate(metadata['variants']):
    name='variant-'+str(variant);run(name,[str(binary),str(shaderfile),str(variant)])
    values=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')];checks=[v for v in values if v['kind']=='check'];samples=[v for v in values if v['kind']=='samples'];assert len(checks)==24
    allchecks.extend(checks)
    for c in checks:
        assert c['variant']==variant and c['generations']==4 and c['passed']==(c['bits_exact'] and c['fp64_worst_ratio']<=1)
        windows=[v for v in samples if all(v[k]==c[k] for k in ('batch','tokens','heads','dim'))]
        if not c['passed']:assert not windows;continue
        assert len(windows)==8
        for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
        assert all(len(v[k])==20 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
        means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
        profiles.append(dict(check=c,label=label,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(profiles=profiles,checks=allchecks,scope=__doc__,full_goal_met=False),indent=2)+'\n').encode());print('Completed original-order vision variant',label,flush=True)
