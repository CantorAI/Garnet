"""Audit fresh original28 gates and regenerate the confined selected dispatch."""
from pathlib import Path
import hashlib,json,math,re,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='shape-selected-two-checks-20261015';assert proof['jobs'][trial]['exit_code']==0
checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-shape-selected-two-body']
assert meta['control_sha256']==proof['library_sha256']['build-resident-kv-inplace']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
assert proof['library_sha256']['build-coalesced-bf16-body']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
for name,sha in proof['library_sha256'].items():assert proof['omitted'][name+'/libgarnet.dylib']['sha256']==sha
preparation=json.loads((root/'preparation.json').read_text())
manifest={name:sha for name,sha in preparation['source_sha256'].items() if 'evidence/source/'+name in proof['files']}
assert len(manifest)>1100
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
sys.path.insert(0,str(root/'source/test2026/metal'))
for name in ('make_shape_selected_two_body','make_two_body','make_coalesced_eight_body'):sys.modules.pop(name,None)
factory=runpy.run_path(str(root/'source/test2026/metal/make_shape_selected_two_body.py'))
operations={'src/backends/metal/metal_context.mm':'context','src/backends/metal/metal_context.h':'header',
    'src/backends/metal/decoder_session.cpp':'session','src/backends/metal/decoder_session.h':'session_header',
    'src/entry/metal_main.cpp':'entry','test2026/metal/run_matched_garnet.py':'worker',
    'test2026/metal/frame_worker_garnet.py':'worker','test2026/metal/coalesced_bf16_body_test.cpp':'primitive'}
changes=preparation['changes'];new='src/backends/metal/coalesced_two_body_shader.h'
assert set(changes)==set(operations)|{new}
for name,operation in operations.items():
    original=root/'original-source'/name
    assert digest(original)==changes[name]['before']
    assert (root/'source'/name).read_bytes()==factory[operation](original.read_text()).encode()
    assert digest(root/'source'/name)==changes[name]['after']
assert (root/'source'/new).read_bytes()==factory['shader']((root/'source/src/backends/metal/coalesced_body_block_shader.h').read_text()).encode()
selection=preparation['selection'];assert len(selection['selected'])==3 and selection['numeric_checks']==960
assert {(p['batch'],p['output_rows'],p['inner'],p['threads']) for p in selection['selected']}==set(factory['TABLE'])
for name,sha in meta['source_sha256'].items():assert digest(root/'source'/name)==sha
assert 'METAL_COALESCED_BF16_BODY' in meta['options'] and 'METAL_MLP_PREFILL_GRAPH' not in meta['options']
template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'
assert digest(template)=='2eb33f027c120dba530aed406ddb60796bfee9484b2e83c10f8a0cc8ab19d2fa'
source=template.read_text();source=source[source.index("stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==28"):]
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
replace("assert stages[:27]==oldstatus and stages[27]['name']=='wide-batch-trained-reference'","assert stages[27]['name']=='wide-batch-trained-reference'")
assert source.count(".relative_to('/Users/zsdx/garnet-metal-investigation')")==3
source=source.replace(".relative_to('/Users/zsdx/garnet-metal-investigation')",".relative_to(preparation['target'])")
replace("assert 'metal_mlp_prefill_graph_reference' in ctext and '0 tests failed out of 8' in ctext","assert '0 tests failed out of 7' in ctext")
replace("assert 'MLP graph pending producer, ownership, abort/recovery and native bound PASS: 64 cases' in last", "assert 'Coalesced body original bits/FP64, policy/tail/fallback, ownership, abort/recovery and prefill checks=104 PASS' in last")
replace('ctests=8,graph_primitive_cases=64,','ctests=7,coalesced_primitive_cases=104,')
replace("    reused_validation_stages=completion['reused_stages'],\n    reused_compile_stages=['configure','build','ctest'],fresh_validation_stages=completion['fresh_stages'],", "    reused_validation_stages=[],reused_compile_stages=[],fresh_validation_stages=[s['name'] for s in stages],")
exec(compile(source,__file__,'exec'))
