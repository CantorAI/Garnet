"""Exact cross-build finite-state/output preservation for saved real probes."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('control');p.add_argument('candidate');p.add_argument('output')
p.add_argument('--prompt-pack-transition',action='store_true')
p.add_argument('--verified-head-transition',action='store_true');a=p.parse_args()
assert not (a.prompt_pack_transition and a.verified_head_transition)
control=json.loads(Path(a.control).read_text());candidate=json.loads(Path(a.candidate).read_text())
control_profile=dict(control['native_profile']);candidate_profile=dict(candidate['native_profile'])
if a.prompt_pack_transition:
    assert control_profile.get('packPrompt',False) is False and candidate_profile.get('packPrompt',False) is True
    control_profile.pop('packPrompt',None);candidate_profile.pop('packPrompt',None)
if a.verified_head_transition:
    assert control_profile.get('reuseVerifiedHead',False) is False and candidate_profile.get('reuseVerifiedHead',False) is True
    control_profile.pop('reuseVerifiedHead',None);candidate_profile.pop('reuseVerifiedHead',None)
keys=('generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids')
signature=lambda r:{k:r[k] for k in keys}
checks={'same_batch':control['batch']==candidate['batch'],
    'same_inputs':control['input_sha256']==candidate['input_sha256'],
    'same_profile':control_profile==candidate_profile,
    'ordinary_states_exact':control['ordinary_state_hashes']==candidate['ordinary_state_hashes'],
    'repeated_states_exact':control['state_hashes']==candidate['state_hashes'],
    'ordinary_outputs_exact':signature(control['ordinary'])==signature(candidate['ordinary']),
    'repeated_outputs_exact':[signature(r) for r in control['trials']]==[signature(r) for r in candidate['trials']]}
result={'control':a.control,'candidate':a.candidate,'checks':checks,'all_passed':all(checks.values()),
    'declared_profile_transition':'packPrompt False to True only' if a.prompt_pack_transition else 'reuseVerifiedHead False to True only' if a.verified_head_transition else None,
    'scope':'Complete finite target-state hashes in identical chunk order and output signatures for this saved real-image probe; not broad independent quality or model speed.'}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
assert result['all_passed'],'Cross-build exact preservation failed; retain both probes and do not promote'
