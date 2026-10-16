"""Exclusive toy mixed-storage alpha gate; no model or performance claim."""
import ast,hashlib,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);output=out
s=root/'source/test2026/metal';statuses=[];background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
nodes=[n for n in ast.parse((s/'run_constant_bf16_probe.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('digest','check','run')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_runner','exec'),globals())
paths=[Path(__file__),s/'make_mps_global_alpha_probe.py',s/'probe_mps_exact_half.mm',s/'compare_vllm_metal.py',s/'run_constant_bf16_probe.py']
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
meta=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
(out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode())
probe=out/'global_alpha_gpu.mm';binary=out/'global_alpha_gpu'
run('generate',[sys.executable,str(s/'make_mps_global_alpha_probe.py'),str(probe)])
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(probe),'-o',str(binary)])
meta.update(probe_sha256=digest(probe),binary_sha256=digest(binary));(out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode())
run('gpu',[str(binary)])
print('Global-alpha toy gate executed successfully; real trained projections not yet tested',flush=True)
