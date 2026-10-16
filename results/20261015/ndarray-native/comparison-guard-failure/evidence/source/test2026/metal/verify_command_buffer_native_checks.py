"""Fresh candidate provenance plus every original28 numerical assertion."""
from pathlib import Path
import hashlib,json,math,re,runpy,sys

base=Path(sys.argv[1]).resolve();root=base/'evidence'
proof=json.loads((root/'manifest.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    path=(base/name).resolve();assert path.is_relative_to(base)
    assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256']
trial='mlp-command-buffer-native-checks-20261015'
assert proof['jobs'][trial]['exit_code']==0
checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text())
assert meta['candidate_sha256']==proof['library_sha256']['build-mlp-command-buffer']=='612af1059c3f86c5d7d5affe3223065b3457c1036bf10464ea3a763672a7e549'
assert meta['control_sha256']==proof['library_sha256']['build-resident-kv-inplace']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
assert proof['library_sha256']['build-coalesced-bf16-body']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
for name,sha in proof['library_sha256'].items():assert proof['omitted'][name+'/libgarnet.dylib']['sha256']==sha
preparation=json.loads((root/'preparation.json').read_text())
assert preparation['prior_primitive_archive_sha256']=='398edcdc8bc6f4b7af2adc53899d6448e93c98b250a33eb7afb02e7d5d19ff5b'
manifest={name:sha for name,sha in preparation['source_sha256'].items() if 'evidence/source/'+name in proof['files']}
assert len(manifest)>1100
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
prior=root/'logs/mlp-command-buffer-chained-20261015/jobs'
prior_meta=json.loads((prior/'metadata.json').read_text());snap=prior/'source-snapshot'
for name,sha in prior_meta['source_sha256'].items():assert digest(snap/name)==sha
factory=runpy.run_path(str(snap/'test2026/metal/make_mlp_command_buffer_context.py'))
changes=preparation['changes']
assert set(changes)=={'src/backends/metal/metal_context.mm','src/backends/metal/mlp_prefill_graph_impl.h','test2026/metal/mlp_prefill_graph_test.cpp'}
for name,operation in (('src/backends/metal/metal_context.mm','context'),('src/backends/metal/mlp_prefill_graph_impl.h','graph'),('test2026/metal/mlp_prefill_graph_test.cpp','primitive')):
    options=dict(chained=True) if operation=='primitive' else {}
    assert digest(snap/name)==changes[name]['before']
    assert (root/'source'/name).read_bytes()==factory[operation]((snap/name).read_text(),**options).encode()
    assert digest(root/'source'/name)==changes[name]['after']==manifest[name]
for name,sha in prior_meta['source_sha256'].items():
    if name not in changes:assert manifest[name]==sha
for name,sha in meta['source_sha256'].items():assert digest(root/'source'/name)==sha
for name in ('METAL_MLP_PREFILL_GRAPH','METAL_COALESCED_BF16_BODY','METAL_INTERLEAVED_SCREEN_INDEX','METAL_SMALL_SCREEN_SEED','METAL_SERIAL_VISION_PREFETCH','METAL_SCREENED_HEAD','METAL_GLOBAL_HALF_ALPHA'):
    assert name in meta['options']
assert all(name not in meta['options'] for name in ('METAL_JOINED_GLOBAL_MLP','METAL_VISION_GLOBAL_ALPHA'))

# Reuse the immutable numerical assertions, not the old job's build/resume
# provenance. This candidate ran fresh from configure through the wide stage.
template=root/'source/test2026/metal/verify_mlp_prefill_native_checks.py'
assert digest(template)=='2eb33f027c120dba530aed406ddb60796bfee9484b2e83c10f8a0cc8ab19d2fa'
source=template.read_text();source=source[source.index("stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==28"):]
needle="assert stages[:27]==oldstatus and stages[27]['name']=='wide-batch-trained-reference'"
assert source.count(needle)==1
source=source.replace(needle,"assert stages[27]['name']=='wide-batch-trained-reference'")
needle=".relative_to('/Users/zsdx/garnet-metal-investigation')"
assert source.count(needle)==3
source=source.replace(needle,".relative_to(preparation['target'])")
needle="assert 'MLP graph pending producer, ownership, abort/recovery and native bound PASS: 64 cases' in last"
assert source.count(needle)==1
source=source.replace(needle,"assert 'MLP command-buffer graph producer/consumer, ownership, abort/recovery and original native bound PASS: 64 cases' in last\nassert 'MLP command-buffer dependent two-graph chains PASS: 64 cases' in last")
needle="    reused_validation_stages=completion['reused_stages'],\n    reused_compile_stages=['configure','build','ctest'],fresh_validation_stages=completion['fresh_stages'],"
assert source.count(needle)==1
source=source.replace(needle,"    reused_validation_stages=[],reused_compile_stages=[],fresh_validation_stages=[s['name'] for s in stages],dependent_graph_chain_cases=64,")
exec(compile(source,__file__,'exec'))
