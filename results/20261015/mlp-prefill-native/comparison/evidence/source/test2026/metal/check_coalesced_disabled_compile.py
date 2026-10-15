"""CPU-only syntax checks for new-flag-off and both-body-flags-off contexts."""
import hashlib,json,re,shlex,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
source=root/'source/src/backends/metal/metal_context.mm';header=source.with_suffix('.h')
flags=(root/'build-coalesced-bf16-body/CMakeFiles/garnet_metal.dir/flags.make').read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')]
assert '-DGARNET_METAL_COALESCED_BF16_BODY=1' in defines
report=dict(scope='CPU Objective-C++ syntax only, no GPU or inference; native serving validation remains separate.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),cases=[])
for name,removed in [('coalesced_off',{'-DGARNET_METAL_COALESCED_BF16_BODY=1'}),('both_body_flags_off',{'-DGARNET_METAL_COALESCED_BF16_BODY=1','-DGARNET_METAL_INTERLEAVED_BF16_BODY=1'})]:
    command=['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-fsyntax-only',*(v for v in defines if v not in removed),str(source)]
    with (out/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    report['cases'].append(dict(name=name,command=command,exit_code=result.returncode));(out/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());assert result.returncode==0,name
print('New flag off and both body flags off: actual native compiler syntax PASS')
