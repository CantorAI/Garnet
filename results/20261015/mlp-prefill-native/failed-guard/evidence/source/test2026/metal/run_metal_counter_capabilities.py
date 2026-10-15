"""Owned guarded capability probe; never concurrent with another inference/probe."""
from pathlib import Path
source=Path(__file__).with_name('run_decode_subsimd_probe.py').read_text()
exec(compile(source[:source.index("native=r/")],__file__,'exec'))
probe=s/'metal_counter_capabilities_probe.mm';binary=output/'metal_counter_capabilities_probe'
(output/'source.mm').write_bytes(probe.read_bytes())
(output/'runner.py').write_bytes(Path(__file__).read_bytes())
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(probe),'-o',str(binary)])
run('capabilities',[str(binary)])
result=json.loads((output/'capabilities.log').read_text())
print(json.dumps(result),flush=True)
