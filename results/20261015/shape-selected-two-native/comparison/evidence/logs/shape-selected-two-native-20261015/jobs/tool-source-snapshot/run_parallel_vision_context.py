"""Exclusive vision contraction component: same FP32 precision and original bounds."""
from pathlib import Path
from make_parallel_vision_context import generate
import statistics
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
paths=[s/'make_parallel_vision_context.py',s/'vision_fused_context_gpu.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',Path(__file__),root/'source/src/backends/metal/metal_context.mm']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},numerical_bounds='Both FP64 and candidate/control original 5e-6+5e-5*abs(reference); FP32 partial sums, altered reduction order; bit identity is observed not required',hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)))
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
shader,cpp=generate(paths[-1].read_text(),(s/'vision_fused_context_gpu.mm').read_text())
shaderfile=out/'parallel-vision.metal';cppfile=out/'parallel_vision_context_gpu.mm';binary=out/'parallel_vision_context_gpu'
shaderfile.write_bytes(shader.encode());cppfile.write_bytes(cpp.encode());metadata.update(shader_sha256=digest(shaderfile),harness_sha256=digest(cppfile))
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(cppfile),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('vision-gpu',[str(binary),str(shaderfile)])
records=[json.loads(v) for v in (out/'vision-gpu.log').read_text().splitlines() if v.startswith('{')]
checks=[v for v in records if v['kind']=='check'];samples=[v for v in records if v['kind']=='samples'];assert len(checks)==24 and len(samples)==192
profiles=[]
for c in checks:
    assert c['generations']==4 and c['fp64_worst_ratio']<=1 and c['control_fp64_worst_ratio']<=1
    windows=[v for v in samples if all(v[k]==c[k] for k in ('batch','tokens','heads','dim'))];assert len(windows)==8
    for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
    assert all(len(v[k])==20 and all(x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
    means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
    profiles.append(dict(check=c,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
(out/'summary.json').write_text(json.dumps(dict(profiles=profiles,scope=__doc__,full_goal_met=False),indent=2)+'\n');print('Passed 24 vision contraction shapes under original numerical bounds',flush=True)
