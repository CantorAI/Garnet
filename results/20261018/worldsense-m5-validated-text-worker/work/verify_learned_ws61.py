"""Post-terminal full learned proof identity and independent original verifier."""
from pathlib import Path
import subprocess, sys, hashlib, json, datetime
w = Path(__file__).resolve().parent.parent / 'worldsense-validated-text-worker'
r = Path('/Users/shawn/CantorAI')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
frozen = json.loads((w / 'learned-input-identities-before.json').read_text())
for key, root in [('parent_source', r / 'out/research/worldsense-validated-text-worker/source'),
                  ('legacy_source', r / 'out/research/worldsense-prefix-cache/source')]:
    assert all(sha(root / rel) == value for rel, value in frozen[key].items())
assert all(sha(Path(p)) == value for p, value in frozen['external'].items())
subprocess.run([sys.executable, str(w / 'verify_correctness_terminal.py'), sys.argv[1]], check=True)
invalid = json.loads((w / 'invalid-image-loaded-regression.json').read_text())
assert invalid['all_six_Galaxy_absent_image_kinds_exact_native_28layer_state_tokens_profile_no_retry']
assert len(invalid['records']) == 6
reference = next(row for row in json.loads((w.parent / 'worldsense-isolated-primary/api-state-guard-off.json').read_text())['records'] if row['model_id'] == 'Qwen3-1.7B')
for row in invalid['records']:
    assert row['state']['states'] == reference['states']
    assert all(row['call'][key] == reference['call'][key] for key in
               ['generated_token_ids', 'text', 'input_prompt_sha256', 'serving_configuration'])
(w / 'learned-independent-audit.json').write_text(json.dumps(dict(
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    controller_session=int(sys.argv[1]), terminal_exit_code=0,
    parent_source_files=len(frozen['parent_source']), legacy_source_files=len(frozen['legacy_source']),
    external_inputs=len(frozen['external']), all_frozen_inputs_unchanged=True,
    original_learned_native_state_token_vision_verifier_pass=True,
    six_invalid_image_outputs_and_states_independently_exact=True,
    scope='Correctness only. Owner-death, config and actual WorldSense gates remain.'), indent=2) + '\n')
print('WS61 full learned proof and frozen-input independent audit PASS')
