"""Audit fresh30 gates and regenerate native transforms/exact-owned-primitive guard."""
from pathlib import Path
import hashlib,json,math,re,runpy,shutil,sys,tempfile
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='ndarray-native-checks-repaired-20261015';assert proof['jobs'][trial]['exit_code']==0
checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text());preparation=json.loads((root/'preparation.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-ndarray-prefill-shape80-guardrepair']
assert proof['library_sha256']['build-coalesced-bf16-body']==preparation['baseline_library_sha256']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
assert meta['control_sha256']==proof['library_sha256']['build-resident-kv-inplace']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
for name,sha in proof['library_sha256'].items():assert proof['omitted'][name+'/libgarnet.dylib']['sha256']==sha
manifest={n:h for n,h in preparation['source_sha256'].items() if 'evidence/source/'+n in proof['files']};assert len(manifest)>1200
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
sys.path.insert(0,str(root/'source/test2026/metal'))
factory=runpy.run_path(str(root/'source/test2026/metal/make_ndarray_native.py'))
operations={'src/backends/metal/metal_context.mm':'context','src/backends/metal/metal_context.h':'header',
    'src/backends/metal/decoder_session.cpp':'session','src/backends/metal/decoder_session.h':'session_header',
    'src/entry/metal_main.cpp':'entry','xModel/qwen35/vl_0_8b/metal_decoder.py':'decoder',
    'test2026/metal/run_matched_garnet.py':'worker','test2026/metal/frame_worker_garnet.py':'worker'}
assert set(preparation['changes'])==set(operations)
for name,operation in operations.items():
    original=root/'original-source'/name;assert digest(original)==preparation['changes'][name]['before']
    assert (root/'source'/name).read_bytes()==factory[operation](original.read_text()).encode()
    assert digest(root/'source'/name)==preparation['changes'][name]['after']
external=json.loads((checks/'external-driver-identities.json').read_text());assert set(external)=={'run_ndarray_native_checks_repaired.py','run_ndarray_native_checks.py','launch_ndarray_native_checks_repaired.py'}
for name,sha in external.items():assert digest(checks/'external-driver-snapshot'/name)==sha
with tempfile.TemporaryDirectory(prefix='garnet-ndarray-audit-') as temp:
    tools=Path(temp)/'tools';tools.mkdir()
    for p in (root/'source/test2026/metal').glob('*.py'):shutil.copyfile(p,tools/p.name)
    for name in external:shutil.copyfile(checks/'external-driver-snapshot'/name,tools/name)
    saved=sys.argv
    try:
        sys.argv=['generate','/unused',str(Path(temp)/'generated'),'--generate-only']
        runpy.run_path(str(tools/'run_ndarray_native_checks_repaired.py'),run_name='__main__')
    finally:sys.argv=saved
    assert (Path(temp)/'generated.complete30.repaired.executed.py').read_bytes()==(checks.parent/'jobs.complete30.repaired.executed.py').read_bytes()
for name,sha in meta['source_sha256'].items():assert digest(root/'source'/name)==sha
assert 'METAL_COALESCED_BF16_BODY' in meta['options'] and 'METAL_MLP_PREFILL_GRAPH' not in meta['options']
template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'
assert digest(template)=='2eb33f027c120dba530aed406ddb60796bfee9484b2e83c10f8a0cc8ab19d2fa'
source=template.read_text();source=source[source.index("stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==28"):]
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace('assert len(stages)==28','assert len(stages)==30')
replace("assert stages[:27]==oldstatus and stages[27]['name']=='wide-batch-trained-reference'","assert stages[29]['name']=='wide-batch-trained-reference'")
replace("assert {v['name'] for v in stages}==names","assert {v['name'] for v in stages}==names|{'ndarray-cache-compile','ndarray-cache'}")
assert source.count(".relative_to('/Users/zsdx/garnet-metal-investigation')")==3
source=source.replace(".relative_to('/Users/zsdx/garnet-metal-investigation')",".relative_to(preparation['target'])")
replace("assert 'metal_mlp_prefill_graph_reference' in ctext and '0 tests failed out of 8' in ctext","assert '0 tests failed out of 7' in ctext")
replace("assert 'MLP graph pending producer, ownership, abort/recovery and native bound PASS: 64 cases' in last","assert 'Coalesced body original bits/FP64, policy/tail/fallback, ownership, abort/recovery and prefill checks=80 PASS' in last")
replace('validation_stages=28,ctests=8,graph_primitive_cases=64,','validation_stages=30,ctests=7,coalesced_primitive_cases=80,ndarray_cache_limit=256,ndarray_cache_fallback_cases=14,')
replace("    reused_validation_stages=completion['reused_stages'],\n    reused_compile_stages=['configure','build','ctest'],fresh_validation_stages=completion['fresh_stages'],","    reused_validation_stages=[],reused_compile_stages=[],fresh_validation_stages=[s['name'] for s in stages],")
needle='Path(sys.argv[2]).write_bytes'
replace(needle,"""cache=next(v for v in stages if v['name']=='ndarray-cache')['accepted_cases'][0]
text=(checks/cache['evidence']/'ndarray-cache.log').read_text()
assert text.strip()=='NDArray native exact/update/cache256/fallback14/cached-after-cap/abort-recovery/shape-fallback PASS'
result.update(regenerated_changes=8,external_driver_files=len(external),regenerated_guard_and_full30_runner=True)
"""+needle)
exec(compile(source,__file__,'exec'))
