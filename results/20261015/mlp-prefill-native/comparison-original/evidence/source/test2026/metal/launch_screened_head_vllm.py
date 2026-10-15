"""Pin separate comparator selection after both native trials pass audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for trial in ('screened-head-native-20261015','screened-head-confirm-native-20261015'):
    assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
for name in ('screened-head-native-verification.json','screened-head-confirm-native-verification.json'):
    audit=json.loads((root/'work'/name).read_text())
    assert audit['original_validation_stages']==19 and audit['additional_trained_head_queries']==51
    assert audit['trained_recovery_cases']==6 and audit['full_output_requests']==72 and len(audit['profiles'])==6
    assert all(p['decode_ratio']>1 for p in audit['profiles'])
manifest=json.loads((root/'screened-head-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
meta=json.loads((root/'logs/screened-head-checks-20261015/jobs/metadata.json').read_text())
sha=hashlib.sha256((root/'build-screened-head/libgarnet.dylib').read_bytes()).hexdigest();assert sha==meta['candidate_sha256']
selection=dict(scope='Fresh additional matched-FP32 screened-head study; no stock BF16 or full-target win claim',models={m:{str(b):'screened_head' for b in (1,2)} for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')},builds=dict(screened_head=dict(directory='build-screened-head',sha256=sha,attention_prefix_enabled=True,global_half_alpha_enabled=True,screened_head_enabled=True)),common_frontend='combined threaded decode and exact-coordinate strip resize',source_manifest=manifest,full_goal_met=False)
path=root/'work/screened-head-next-comparator-selection.json';raw=(json.dumps(selection,indent=2)+'\n').encode()
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw)
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_screened_head_vllm.py')
exec(compile(source,__file__,'exec'))
