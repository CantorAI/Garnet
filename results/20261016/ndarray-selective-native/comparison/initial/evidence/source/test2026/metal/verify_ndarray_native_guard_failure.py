"""Narrow independent provenance audit of the archived self-guard interruption."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base)
    assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='ndarray-native-checks-20261015';assert proof['jobs'][trial]['exit_code']==1
prep=json.loads((root/'preparation.json').read_text());manifest={n:h for n,h in prep['source_sha256'].items() if 'evidence/source/'+n in proof['files']}
assert len(manifest)>1200
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
sys.path.insert(0,str(root/'source/test2026/metal'))
factory=runpy.run_path(str(root/'source/test2026/metal/make_ndarray_native.py'))
operations={'src/backends/metal/metal_context.mm':'context','src/backends/metal/metal_context.h':'header',
    'src/backends/metal/decoder_session.cpp':'session','src/backends/metal/decoder_session.h':'session_header',
    'src/entry/metal_main.cpp':'entry','xModel/qwen35/vl_0_8b/metal_decoder.py':'decoder',
    'test2026/metal/run_matched_garnet.py':'worker','test2026/metal/frame_worker_garnet.py':'worker'}
assert set(prep['changes'])==set(operations)
for name,operation in operations.items():
    original=root/'original-source'/name;assert digest(original)==prep['changes'][name]['before']
    assert (root/'source'/name).read_bytes()==factory[operation](original.read_text()).encode()
    assert digest(root/'source'/name)==prep['changes'][name]['after']
out=root/'logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text())
assert proof['library_sha256']['build-ndarray-prefill-shape80']==meta['candidate_sha256']
assert proof['library_sha256']['build-coalesced-bf16-body']==prep['baseline_library_sha256']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
for name,sha in meta['source_sha256'].items():assert digest(root/'source'/name)==sha
stages=json.loads((out/'statuses.json').read_text());assert [v['name'] for v in stages]==['configure','build','ndarray-cache-compile']
assert all(v['exit_code']==0 and v['interruption'] is None for v in stages)
for v in stages:
    assert len(v['accepted_cases'])==1
    q=v['accepted_cases'][0];folder=out/q['evidence'];assert q['status']['exit_code']==0 and q['status']['interruption'] is None
    assert json.loads((folder/(v['name']+'-monitor.json')).read_text())
attempts=list((out/'attempts/ndarray-cache').glob('case0-attempt*/statuses.json'));assert attempts
for path in attempts:
    rows=json.loads(path.read_text());assert len(rows)==1 and rows[0]['name']=='ndarray-cache'
    assert rows[0]['exit_code']==1 and rows[0]['interruption'].startswith('Other GPU probe PID ')
    assert Path(rows[0]['command'][0]).name=='ndarray_prefill_cache_gpu'
stop=json.loads((out.parent/'owned-stop.json').read_text());assert stop['pid']==81353 and stop['signal']=='SIGINT' and stop['foreign_processes_untouched']
result=dict(verified_files=len(proof['files']),regenerated_changes=len(operations),accepted_setup_stages=3,interrupted_primitive_attempts=len(attempts),
    source_manifest_files=len(manifest),candidate_library_sha256=meta['candidate_sha256'],
    scope='Build/source/guard failure only. No accepted primitive, trained validation, full inference or performance claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
