"""Fresh original28 stages plus actual-library NDArray cache/fallback/abort primitive."""
from pathlib import Path
import re,sys
p=Path(__file__).with_name('run_guarded_coalesced_body_native_checks.py');ns={'__file__':str(p)};wrapper=p.read_text()
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
wrapper=ns['source'];marker="exec(compile(source,str(destination),'exec'))";assert wrapper.count(marker)==1
before,after=wrapper.split(marker);exec(compile(before,str(p),'exec'),ns);source=ns['source']+after
source=source.replace('build-coalesced-bf16-body','build-ndarray-prefill-shape80')
needle="run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])";assert source.count(needle)==1
source=source.replace(needle,"""run('ndarray-cache-compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(r/'source/src/backends/metal'),str(s/'ndarray_prefill_cache_gpu.mm'),str(build/'libgarnet_metal.a'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(output/'ndarray_prefill_cache_gpu')])
run('ndarray-cache',[str(output/'ndarray_prefill_cache_gpu')])
"""+needle)
source=re.sub(r'assert len\(statuses\)==(\d+)',lambda m:'assert len(statuses)=='+str(int(m[1])+2),source)
destination=Path(sys.argv[2]).with_suffix('.complete30.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
