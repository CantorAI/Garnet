"""Original resource preparation/dispatch semantics, explicit fresh prerequisites."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved-text';old=b/'work/worldsense-isolated-primary';slug='worldsense-sole-interleaved-text'
assert json.loads((w/'learned-independent-audit.json').read_text())['all_original_model_word_state_token_vision_gates_pass']
assert json.loads((w/'functional-smoke/functional-verification.json').read_text())['actualWorldSense_Galaxy_native_candidate_functional_pass']
t=(old/'prepare_cohort.py').read_text().replace('worldsense-isolated-primary',slug)
start=t.index('assert all(v==0');end=t.index("subprocess.run(['python3',str(w/'freeze.py')],check=True)")
t=t[:start]+'''assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
assert json.loads((w/'native-word-comparison.json').read_text())['all_six_builds_FP32_words_identical']
assert json.loads((w/'sole-word-comparison.json').read_text())['all_six_builds_FP32_words_identical']
assert json.loads((w/'correctness-terminal.json').read_text())['all_three_actual_model_modes_and_five_readers_exact']
assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert json.loads((w/'learned-independent-audit.json').read_text())['all_original_model_word_state_token_vision_gates_pass']
assert json.loads((w/'learned-independent-audit.json').read_text())['all197_actual_text_weights_GPU_original_FP32_words']==1720451072
assert json.loads((w/'learned-independent-audit.json').read_text())['three_actual_direct_prefill_tile_readers_each_FP32_words']==1720451072
assert json.loads((w/'admission-complete.json').read_text())['twentythree_expected_config_rejections']
assert json.loads((w/'acceptance-gate-audit.json').read_text())['all_five_original_golden_numeric_metric_gate_AST_unchanged']
assert json.loads((w/'acceptance-gate-audit.json').read_text())['full_route_mutation_audit_pass']
assert json.loads((w/'functional-smoke/functional-verification.json').read_text())['actualWorldSense_Galaxy_native_candidate_functional_pass']
assert json.loads((w/'source-inverse-proof.json').read_text())['original_shader_literals_and_63_decode_graph_session_xModel_image_CUDA_original_tests_unchanged']
'''+t[end:]
ast.parse(t);(w/'prepare_cohort.py').write_text(t);(w/'run_resource.py').write_text((old/'run_resource.py').read_text())
# Dispatch wrapper is saved before preparation freezes the entire driver tree.
driver='''from pathlib import Path
import subprocess,sys,json,hashlib
w=Path(__file__).resolve().parent
assert json.loads((w/'cohort-preparation-complete.json').read_text())['all_prerequisites_checked']
identity=json.loads((w/'driver-identities-before-dispatch.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['files'].items())
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['files'].items())
 subprocess.run([sys.executable,str(w/'dependency_closure.py'),'resource'],check=True)
 subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
'''
ast.parse(driver);(w/'dispatch_prepared_resource.py').write_text(driver)
print('Original eight-call resource controller and thresholds prepared with fresh explicit prerequisites; no dispatch')
