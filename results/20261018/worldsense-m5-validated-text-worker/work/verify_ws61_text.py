"""Post-terminal independent audit of the first WS61 learned text proof."""
from pathlib import Path
import json, hashlib, datetime, sys

w = Path(__file__).resolve().parent.parent / 'worldsense-validated-text-worker'
r = Path('/Users/shawn/CantorAI')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

frozen = json.loads((w / 'first-model-identities-before.json').read_text())
roots = {'parent_source': r / 'out/research/worldsense-validated-text-worker/source',
         'legacy_source': r / 'out/research/worldsense-prefix-cache/source'}
checked = {}
for kind, root in roots.items():
    for rel, expected in frozen[kind].items():
        assert sha(root / rel) == expected, (kind, rel)
    checked[kind] = len(frozen[kind])
for path, expected in frozen['external'].items():
    assert sha(Path(path)) == expected, path
checked['external'] = len(frozen['external'])
assert all(json.loads((w / 'text-model-guards-result.json').read_text()).values())
refs = {row['prompt_index']: row for row in json.loads(
    (w.parent / 'worldsense-prefix-cache/prefix-model-guard.json').read_text())['records']}
roles = {}
for mode in ['on', 'primary', 'fault']:
    d = json.loads((w / ('text-model-guard-' + mode + '.json')).read_text())
    assert d['all28text_state_bits_tokens_prefix_capacity_error_profile_close_reload_policy_clock_cohort_memory_pass']
    assert len(d['first_three']) == 3
    for row, index in zip(d['first_three'], [0, 1, 0]):
        assert row['prompt_index'] == index and len(row['states']) == 28
        assert row['states'] == refs[index]['states']
        call = row['call']
        assert call['generated_token_ids'] == refs[index]['generated_token_ids']
        assert call['input_prompt_sha256'] == refs[index]['prompt_sha256']
        child = call['timings']['native_gateway']['child']
        assert child['model_backend']['library_sha256'] == 'dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
        assert child['raw_counter_policy'] == 'pinned_ws10_structural_absence'
    for key in ['additional_prefix_disabled_call', 'policy_enabled_call']:
        assert d[key]['generated_token_ids'] == refs[0]['generated_token_ids']
    roles[mode] = {'exact_initial_calls': 3, 'all28_states_exact': True,
                   'reload_and_policy_tokens_exact': True}
result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              observed_terminal_session=int(sys.argv[1]), observed_exit_code=0,
              frozen_identities_unchanged=checked, roles=roles,
              scope='Text correctness only. No VLM model proof or WorldSense speed claim.')
(w / 'text-independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print('WS61 three-role text outputs/states/reload/policy and complete first-model identities PASS')
