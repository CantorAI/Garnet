"""Require two complete native audits before freezing a fresh comparator policy."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve()
for name in ('coalesced-body-native-verification.json','coalesced-body-native-confirm-verification.json'):
    audit=json.loads((root/'work'/name).read_text())
    assert audit['full_output_requests']==72 and audit['actual_coalesced'] and audit['coalesced_threads']==64
    assert audit['wide_reference_profiles']==4 and audit['wide_carried_steps']==12
joined=json.loads((root/'work/coalesced-body-native-combined-verification.json').read_text())
assert joined['full_output_requests']==144 and joined['identical_assets_libraries_sources_and_cases']
assert all(p['decode_faster_both'] for p in joined['profiles'])
for name in ('coalesced-body-native-checks-20261015','coalesced-body-native-20261015','coalesced-body-native-confirm-20261015','combined-graph-prefill-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
manifest=json.loads((root/'logs/coalesced-body-native-20261015/source-identities.json').read_text())
for name in ('compare_coalesced_body_vllm.py','run_coalesced_body_vllm.py','launch_coalesced_body_vllm.py','archive_coalesced_body_vllm.py','verify_coalesced_body_vllm.py'):
    rel='test2026/metal/'+name;manifest[rel]=hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest()
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
sha=hashlib.sha256((root/'build-coalesced-bf16-body/libgarnet.dylib').read_bytes()).hexdigest()
assert sha=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
selection=dict(scope='Fresh matched FP32 cohort of the lossless BF16 body candidate; extra storage/init costs retained; no stock BF16 claim',models={m:{str(b):'coalesced_body' for b in (1,2)} for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')},builds=dict(coalesced_body=dict(directory='build-coalesced-bf16-body',sha256=sha,attention_prefix_enabled=True,global_half_alpha_enabled=True,screened_head_enabled=True)),source_manifest=manifest,common_frontend='combined threaded decode and exact-coordinate strip resize',vl_b2_internal_prefill_plan=[80],full_goal_met=False)
path=root/'work/coalesced-body-next-comparator-selection.json';raw=(json.dumps(selection,indent=2)+'\n').encode()
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw)
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_coalesced_body_vllm.py')
exec(compile(source,__file__,'exec'))
