"""Freeze a separate comparator policy after both full native output audits."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('wide-batch-prefill-verification.json','wide-batch-prefill-confirm-verification.json'):
    audit=json.loads((root/'work'/name).read_text());assert audit['full_output_requests']==72 and audit['wide_reference_profiles']==4 and audit['wide_carried_steps']==12
for name in ('wide-batch-prefill-checks-20261015','wide-batch-prefill-native-20261015','wide-batch-prefill-native-confirm-20261015','interleaved-bf16-body-20261015','interleaved-bf16-body-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
manifest=json.loads((root/'wide-batch-prefill-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
meta=json.loads((root/'logs/wide-batch-prefill-checks-20261015/jobs/metadata.json').read_text());sha=hashlib.sha256((root/'build-wide-batch-prefill/libgarnet.dylib').read_bytes()).hexdigest();assert sha==meta['candidate_sha256']
selection=dict(scope='Fresh additional matched FP32 comparison of the validated latency bundle; no stock BF16 claim',models={m:{str(b):'wide_batch' for b in (1,2)} for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')},builds=dict(wide_batch=dict(directory='build-wide-batch-prefill',sha256=sha,attention_prefix_enabled=True,global_half_alpha_enabled=True,screened_head_enabled=True)),source_manifest=manifest,common_frontend='combined threaded decode and exact-coordinate strip resize',vl_b2_internal_prefill_plan=[80],full_goal_met=False)
path=root/'work/wide-batch-next-comparator-selection.json';raw=(json.dumps(selection,indent=2)+'\n').encode()
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw)
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_wide_batch_vllm.py')
exec(compile(source,__file__,'exec'))
