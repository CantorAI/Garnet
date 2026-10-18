"""Bounded native no-model feasibility, preserving old selected source/library."""
from pathlib import Path
import subprocess,json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-prefix-cache/source'
manifest=json.loads((w.parent/'worldsense-primary-u32-integration/control-source-before-dispatch.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(manifest)==1706 and all(sha(s/k)==v for k,v in manifest.items())
library=r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'
assert sha(library)=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
n=r/'out/build/macos-arm64-release/bin'
paths=[library,n/'xlang3',n/'libxlang3_runtime.dylib',w/'native_module_probe.py',w/'mock-adapter/metal_serving.py',w/'feasibility.py']
identity=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_source=manifest,files={str(p):sha(p)for p in paths},command=[str(n/'xlang3'),'-B',str(w/'native_module_probe.py')],scope='Before native no-model dispatch probe; no Garnet source or binary copy/edit/build.')
(w/'feasibility-identities-before.json').write_text(json.dumps(identity,indent=2)+'\n')
with(w/'native-module-probe.log').open('w')as log:
 result=subprocess.run(identity['command'],stdout=log,stderr=subprocess.STDOUT,timeout=60)
(w/'feasibility-observed-terminal.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode),indent=2)+'\n')
assert all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(p))==v for p,v in identity['files'].items())
assert result.returncode==0,'Native dispatch probe failed; preserve logs and reconsider before implementation'
probe=json.loads((w/'native-module-probe-result.json').read_text())
assert probe['native_package_status_and_error_callbacks_reach_preimported_adapter']
(w/'feasibility-result.json').write_text(json.dumps(dict(old1706_source_library_runtime_probe_drivers_unchanged=True,native_preimported_adapter_resolution_pass=True,no_model_inference=True,no_worker_or_performance_admission=True,scope='This supports adapter feasibility only. Real native clocks/memory/ownership/typed arguments/full numerical and actualWorldSense gates remain required.'),indent=2)+'\n')
print('WS61 native adapter-resolution feasibility PASS; exact selected WS10 source/library unchanged; no model/speed result',flush=True)
