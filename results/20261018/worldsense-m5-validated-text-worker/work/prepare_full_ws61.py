"""Adapt unchanged WS58 mathematical guards to the explicit WS61 provenance."""
from pathlib import Path
import hashlib, json, ast

root = Path(__file__).resolve().parent.parent
old = root / 'worldsense-native-text-worker'
w = root / 'worldsense-validated-text-worker'
assert (w / 'text-independent-audit.json').is_file()
names = ['build_corpus.py', 'build_direct_corpus.py', 'make_resident_fixtures.py',
         'resident_controller.py', 'resident_guard.py', 'linear_guard.py',
         'vision_grid_guard.py', 'api_state_guard.py', 'full_correctness.py',
         'verify_correctness_terminal.py']
records = {}
for name in names:
    original = (old / name).read_text()
    text = original.replace('worldsense-native-text-worker', 'worldsense-validated-text-worker')
    text = text.replace('source1723', 'source1726').replace('len(manifest)==1723', 'len(manifest)==1726')
    if name == 'api_state_guard.py':
        text = text.replace('import garnet,metal_serving',
                            'import garnet,metal_serving\nfrom validated_call_guard import validated_call')
        target = "assert call['status']=='ok'\n"
        assert text.count(target) == 1
        text = text.replace(target, target + ' if remote:validated_call(call,mode)\n')
    ast.parse(text)
    (w / name).write_text(text)
    records[name] = dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),
                         current_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         changes='Directory/source-count identities; api_state_guard adds native dual-provenance admission only. All numeric/state/token/golden assertions retained.')
for name in ['primary_corpus_test.cpp', 'direct_float_corpus_test.cpp', 'direct-upload-selection.json']:
    (w / name).write_bytes((old / name).read_bytes())
    records[name] = dict(seed_sha256=hashlib.sha256((old / name).read_bytes()).hexdigest(), byte_exact_seed=True)
# Undo only the documented mechanical changes and added admission to prove all
# original numerical/control/golden expressions survive byte-for-byte.
for name in names:
    text = (w / name).read_text().replace('worldsense-validated-text-worker', 'worldsense-native-text-worker')
    text = text.replace('source1726', 'source1723').replace('len(manifest)==1726', 'len(manifest)==1723')
    if name == 'api_state_guard.py':
        text = text.replace('\nfrom validated_call_guard import validated_call', '')
        text = text.replace(' if remote:validated_call(call,mode)\n', '')
    assert text == (old / name).read_text(), name
(w / 'full-guard-adaptation-proof.json').write_text(json.dumps(records, indent=2) + '\n')
opt = Path(__file__).resolve().parent
ownership = {}
for name in ['owner_death', 'owner_parent', 'post_owner_recovery']:
    seed = opt / ('ws58_' + name + '.py')
    original = seed.read_text()
    text = original.replace('worldsense-native-text-worker', 'worldsense-validated-text-worker')
    text = text.replace('ws58_owner_parent.py', 'ws61_owner_parent.py')
    if name == 'post_owner_recovery':
        text = text.replace('import garnet\n',
                            'import garnet\nsys.path.insert(0,str(w))\nfrom validated_call_guard import validated_call\n')
        target = "save();assert result['call']['status']=='ok';result['state']="
        assert target in text
        text = text.replace(target, "save();assert result['call']['status']=='ok';validated_call(result['call'],'on');result['state']=")
    ast.parse(text)
    target = opt / ('ws61_' + name + '.py')
    target.write_text(text)
    inverse = text.replace('worldsense-validated-text-worker', 'worldsense-native-text-worker')
    inverse = inverse.replace('ws61_owner_parent.py', 'ws58_owner_parent.py')
    if name == 'post_owner_recovery':
        inverse = inverse.replace('\nsys.path.insert(0,str(w))\nfrom validated_call_guard import validated_call', '')
        inverse = inverse.replace(";validated_call(result['call'],'on')", '')
    assert inverse == original, name
    ownership[name] = dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),
                           current_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           original_active_native_stack_and_ownership_assertions_inverse_exact=True)
(w / 'ownership-guard-adaptation-proof.json').write_text(json.dumps(ownership, indent=2) + '\n')
print('WS61 full learned/native guards prepared; inverse-exact original numeric/golden assertions')
