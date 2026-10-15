"""Audit actual block dispatches, original full28 gates and complete generated outputs."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_interleaved_body_native.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-interleaved-bf16-body','build-coalesced-bf16-body'),('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015'),('interleaved-body-native-20261015','coalesced-body-native-20261015'),('interleaved-body-native-original-source-20261015','coalesced-body-native-original-source-20261015')]:source=source.replace(old,new)
source=source.replace("dependencies['pid']==49634","dependencies['pid']==json.loads((checks.parent/'pid.json').read_text())['pid']")
needle="            assert row['interleaved_body_enabled']==(policy=='candidate')";assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert row['coalesced_block_enabled']==(policy=='candidate')
            compressed=row['coalesced_block_counts'];assert len(compressed)==row['decode_steps_after_first']+1 and len(compressed)>1
            assert compressed[0]==0 and all((v>0)==(policy=='candidate') for v in compressed[1:]) and len(set(compressed[1:]))==1
''')
needle='Path(sys.argv[2]).write_bytes';assert source.count(needle)==1
source=source.replace(needle,'''assert 'metal_coalesced_bf16_body_reference' in ctext
assert 'METAL_COALESCED_BF16_BODY' in json.loads((checks/'metadata.json').read_text())['options']
import re
shader=re.compile(r'R"METAL\\((.*?)\\)METAL"',re.S)
assert shader.search((root/'source/src/backends/metal/coalesced_body_block_shader.h').read_text()).group(1)==shader.search((root/'source/test2026/metal/coalesced_body_block_shader.h').read_text()).group(1)
for p in profiles:
    directory=native/Path(p['evidence']).name;meta=json.loads((directory/'metadata.json').read_text())
    assert meta['coalesced_body']==dict(threads=64,lossless_storage=True,fp32_arithmetic=True,original_raw_fallback=True,model_workloads_unchanged=True)
result.update(actual_coalesced=True,coalesced_threads=64,native_primitive_cases=80)
off=json.loads((root/'logs/coalesced-native-disabled-compile-20261015/report.json').read_text())
assert off['source_sha256']==digest(root/'source/src/backends/metal/metal_context.mm') and off['header_sha256']==digest(root/'source/src/backends/metal/metal_context.h')
assert len(off['cases'])==2 and {v['name'] for v in off['cases']}=={'coalesced_off','both_body_flags_off'}
for case in off['cases']:
    assert case['exit_code']==0 and '-fsyntax-only' in case['command'] and '-DGARNET_METAL_COALESCED_BF16_BODY=1' not in case['command']
    assert ('-DGARNET_METAL_INTERLEAVED_BF16_BODY=1' in case['command'])==(case['name']=='coalesced_off')
result.update(disabled_context_syntax_cases=2)
'''+needle)
exec(compile(source,__file__,'exec'))
