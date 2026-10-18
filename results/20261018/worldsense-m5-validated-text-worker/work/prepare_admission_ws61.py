"""Preserve WS58 configuration guards and append four validated-worker guards."""
from pathlib import Path
import ast, hashlib, json
opt = Path(__file__).resolve().parent
w = opt.parent / 'worldsense-validated-text-worker'
seed = (opt / 'ws58_admission.py').read_text()
text = seed.replace('worldsense-native-text-worker', 'worldsense-validated-text-worker')
old = "assert json.loads((w/'reproof-complete.json').read_text())['full_native_text_VLM_corpus_vision_lifecycle_and_parent_telemetry_reproof_pass']"
new = "assert json.loads((w/'learned-independent-audit.json').read_text())['original_learned_native_state_token_vision_verifier_pass']"
assert text.count(old) == 1
text = text.replace(old, new)
needle = 'for name,flags,expected in cases:\n'
extra = '''
validated_common=['-DGARNET_METAL_NATIVE_TEXT_WORKER=ON','-DGARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW=ON','-DGARNET_METAL_TEXT_WORKER_RAW_ROOT='+str(r/'out/build/garnet-metal-worldsense-prefix-cache'),'-DGARNET_METAL_TEXT_WORKER_EXECUTABLE='+str(r/'out/build/macos-arm64-release/bin/xlang3'),'-DGARNET_METAL_VALIDATED_RAW_SOURCE_ROOT='+str(r/'out/research/worldsense-prefix-cache/source')]
cases += [
 ('validated-no-worker',['-DGARNET_METAL_NATIVE_TEXT_WORKER=OFF','-DGARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW=ON'],'Validated RAW worker requires owned native worker and Metal xModel API'),
 ('validated-no-xmodel',validated_common+['-DGARNET_METAL_XMODEL=OFF'],'Native text worker requires Metal xModel API'),
 ('validated-missing-source',validated_common+['-DGARNET_METAL_VALIDATED_RAW_SOURCE_ROOT=/nonexistent/ws61-source'],'Validated RAW worker requires its immutable original source closure'),
 ('validated-wrong-library',validated_common+['-DGARNET_METAL_TEXT_WORKER_RAW_ROOT='+str(r/'out/build/garnet-metal-worldsense-validated-text-worker-off')],'Validated RAW worker library identity mismatch')]
'''
assert text.count(needle) == 1
text = text.replace(needle, extra + needle)
text = text.replace('nine_expected_config_rejections=True', 'thirteen_expected_config_rejections=True')
text = text.replace('Original6/new3 invalid config guards', 'Original9/new4 validated-worker invalid config guards')
text = text.replace('native-source/build admission; never starts model timing', 'native-source/build admission; never starts model timing')
ast.parse(text)
(opt / 'ws61_admission.py').write_text(text)
inverse = text.replace(extra, '').replace('worldsense-validated-text-worker', 'worldsense-native-text-worker')
inverse = inverse.replace(new, old).replace('thirteen_expected_config_rejections=True', 'nine_expected_config_rejections=True')
inverse = inverse.replace('Original9/new4 validated-worker invalid config guards', 'Original6/new3 invalid config guards')
assert inverse == seed
(w / 'config-guard-adaptation-proof.json').write_text(json.dumps(dict(
    original_nine_config_cases_and_assertions_inverse_exact=True,
    new_validated_worker_negative_cases=4,
    seed_sha256=hashlib.sha256(seed.encode()).hexdigest(),
    current_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2) + '\n')
print('WS61 config guard prepared: original nine plus four explicit validated-worker failures')
