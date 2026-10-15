"""Build/run the isolated exact codec; preserve commands and exclusive GPU use.

Usage: Python script investigation_root evidence_directory [--small]
No inference integration, generation-speed or vLLM advantage is claimed.
"""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path

root=Path(sys.argv[1]);output=Path(sys.argv[2]);small='--small' in sys.argv[3:]
single='--single-column' in sys.argv[3:]
reuse=Path(sys.argv[sys.argv.index('--reuse-assets')+1]).resolve() if '--reuse-assets' in sys.argv else None
output.mkdir(parents=True,exist_ok=False);scripts=root/'source/test2026/metal';statuses=[]

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

import ast
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
strict_guard=guard

def guard(label):
    strict_guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('/lossless_block_gpu_probe','/decode_single_output_gpu','/parallel_delta_gpu','/lossless_block_gpu_probe','/prefill_tile_gpu')):
            raise RuntimeError('Other GPU probe PID '+fields[0])
    (output/(label+'-processes.txt')).write_text(snapshot())

def run(name,command):
    begin=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode,wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if result.returncode:raise RuntimeError(name+' failed; see retained log')

native=root/'source/src/backends/metal/metal_context.mm'
text=native.read_text();begin=text.index('struct BF16Params');end=text.index('// BF16 weights are expanded',begin)
control=text[begin:end];shader=output/'combined.metal'
shader.write_text('#include <metal_stdlib>\nusing namespace metal;\n'+control+(scripts/'lossless_block_gpu_probe.metal').read_text())
metadata=dict(scope='Exact GPU codec and isolated matrix-vector feasibility only; not full-model inference',
    codec='whole-block-fallback',
    packed_columns_per_simd=1 if single else 4,
    reuse_assets=str(reuse) if reuse else None,
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
    source_sha256={str(path.name):digest(path) for path in (Path(__file__),native,scripts/'lossless_block_gpu_probe.mm',scripts/'lossless_block_gpu_probe.metal',scripts/'prepare_lossless_block_gpu_probe.py',scripts/'probe_lossless_block_bf16_codec.py')},
    native_control_kernel_sha256=hashlib.sha256(control.encode()).hexdigest(),small=small)
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
binary=output/'lossless_block_gpu_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(scripts/'lossless_block_gpu_probe.mm'),'-o',str(binary)])
summary=[]
for model in ('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    guard(model+'-before')
    assets=output/(model+'-assets')
    if reuse:
        origin=reuse/(model+'-assets');assets.mkdir()
        manifest=json.loads((origin/'manifest.json').read_text())
        assert manifest['checkpoint_sha256']==digest(root/'models'/model/'model.safetensors') and manifest['small']==small
        for path in origin.rglob('*.json'):
            target=assets/path.relative_to(origin);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        for config_path in origin.glob('*/case.json'):
            config=json.loads(config_path.read_text());assert config['codec']=='whole-block-fallback'
            for name,expected in config['sha256'].items():assert digest(config_path.parent/name)==expected
        (assets/'reuse.json').write_text(json.dumps(dict(origin=str(origin),every_input_digest_verified=True),indent=2)+'\n')
        gpu_assets=origin
    else:
        run(model+'-prepare',[sys.executable,str(scripts/'prepare_lossless_block_gpu_probe.py'),str(root/'models'/model/'model.safetensors'),str(assets)]+(['--small'] if small else []))
        gpu_assets=assets
    guard(model+'-gpu-before');run(model+'-gpu',[str(binary),str(shader),str(gpu_assets)]+(['single'] if single else []));guard(model+'-after')
    rows=[json.loads(line) for line in (output/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    roundtrips=[row for row in rows if row['kind']=='roundtrip'];checks=[row for row in rows if row['kind']=='projection_check']
    assert len(roundtrips)==4 and len(checks)==4
    for case in ('matrix-0','matrix-1'):
        for batch in (1,2):
            times=[row for row in rows if row['kind']=='timing' and row['case']==case and row['batch']==batch]
            assert len(times)==16
            raw=statistics.median(row['gpu_ms_median'] for row in times if row['policy']=='raw')
            packed=statistics.median(row['gpu_ms_median'] for row in times if row['policy']=='packed')
            config=json.loads((assets/case/'case.json').read_text())
            summary.append(dict(model=model,case=case,batch=batch,rows=config['rows'],inner=config['inner'],storage_ratio=config['storage_ratio'],
                raw_gpu_ms=raw,packed_gpu_ms=packed,isolated_gpu_speedup=raw/packed,
                gpu_weight_bits_exact=True,projection_check=next(row for row in checks if row['case']==case and row['batch']==batch)))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=summary),indent=2)+'\n')
print(json.dumps(dict(scope=metadata['scope'],results=summary),indent=2))
