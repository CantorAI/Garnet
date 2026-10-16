"""Actual native-context mixed-storage padded-row matrix component; no model TPS."""
from pathlib import Path
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index("native=root/")],__file__,'exec'),ns);globals().update(ns)
import math,statistics
build=root/'build-screened-head';library=build/'libgarnet_metal.a'
assert digest(build/'libgarnet.dylib')=='2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23'
for trial in ('screened-prefill-reset-native-20261015','screened-prefill-reset-confirm-native-20261015'):assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
paths=[Path(__file__),s/'padded_mps_body_gpu.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',root/'source/src/backends/metal/metal_context.h',root/'source/src/backends/metal/metal_context.mm',root/'source/src/backends/metal/exact_half_codec.h']
metadata=dict(scope=__doc__,limitations='Component only; repeatedly resident individual matrices, padding/copy into resident inputs outside timing, no full-model or vLLM claim',numerical_bounds='Unchanged FP64 gamma(ceil(K/32)+32)*sumabs+1e-7 and original-control 5e-6+5e-5*abs(control)',library_sha256=digest(library),source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},versions=dict(python=sys.version,compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
binary=out/'padded_mps_body_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(root/'source/src/backends/metal'),str(s/'padded_mps_body_gpu.mm'),str(library),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
metadata['binary_sha256']=digest(binary);profiles=[];all_checks=[];skipped=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors'
    assert digest(checkpoint)==info['checkpoint_sha256']
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        header_bytes=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+header_bytes]);start=8+header_bytes
        for c in info['cases']:
            raw=assets/c['name']/'raw.bin';assert digest(raw)==c['sha256']['raw.bin']
            tensor=header[c['tensor']];assert tensor['dtype']=='BF16' and tensor['shape']==[c['rows'],c['inner']]
            lo,hi=tensor['data_offsets'];assert hashlib.sha256(mapping[start+lo:start+hi]).hexdigest()==c['sha256']['raw.bin']
    metadata['assets'][model]=info;(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(assets)])
    values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[v for v in values if v['kind']=='check'];all_checks.extend(dict(model=model,**v) for v in checks);skipped.extend(dict(model=model,**v) for v in values if v['kind']=='skipped')
    for c in info['cases']:
        if any(v['matrix']==c['name'] for v in values if v['kind']=='skipped'):continue
        for batch in (1,2):
            for physical in (4,8,16):
                relevant=[v for v in checks if v['matrix']==c['name'] and v['batch']==batch and v['physical_rows']==physical];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
                windows=[v for v in values if v['kind']=='samples' and v['matrix']==c['name'] and v['batch']==batch and v['physical_rows']==physical]
                if not all(v['passed'] for v in relevant):assert not windows;continue
                assert len(windows)==8
                for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
                assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
                means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
                profiles.append(dict(model=model,matrix=c['name'],tensor=c['tensor'],batch=batch,physical_rows=physical,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,skipped=skipped,full_goal_met=False),indent=2)+'\n').encode());print('Completed native padded-row component',model,flush=True)
