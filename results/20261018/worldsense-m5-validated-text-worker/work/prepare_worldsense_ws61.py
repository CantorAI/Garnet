"""Prepare original WorldSense gates for the explicit WS61 dual backend."""
from pathlib import Path
import ast, json, hashlib
opt = Path(__file__).resolve().parent
root = opt.parent
w = root / 'worldsense-validated-text-worker'
seed = root / 'worldsense-native-text-worker'
original = root / 'worldsense-isolated-primary'
assert json.loads((w / 'admission-complete.json').read_text())['thirteen_expected_config_rejections']
route = (opt / 'ws61_native_route.py').read_text()
route = route[route.index('def native_route('):].rstrip() + '\n'
rels = ['analyze_resource.py', 'speed/analyze.py', 'primary-speed/analyze.py',
        'production-speed/analyze.py', 'confirmation/analyze.py']
class Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        return None if node.name == 'native_route' else self.generic_visit(node)
checks = {}
for rel in rels:
    text = (original / rel).read_text().replace('worldsense-isolated-primary', 'worldsense-validated-text-worker')
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_route')
    replacement = route
    if rel == 'production-speed/analyze.py':
        segment = ast.get_source_segment(text, fn)
        prefix = segment[segment.index(' if not enabled:'):segment.index(' packed=')]
        replacement = replacement.replace(' import math\n', prefix + ' import math\n', 1)
    lines = text.splitlines(keepends=True)
    text = ''.join(lines[:fn.lineno-1]) + replacement + ''.join(lines[fn.end_lineno:])
    # The child is deliberately loaded from a separate serving overlay. Keep
    # every original assertion; change only the exact source-path selector.
    old_path = "source/'runtime/metal_serving.py'"
    if rel == 'production-speed/analyze.py':
        old_path = "(source if expected else Path(environment['control_source']))/'runtime/metal_serving.py'"
        new_path = "(source if expected else Path(environment['control_source']))/('runtime/validated_ws10/metal_serving.py'if expected and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')"
    else:
        condition = "mode=='on'" if rel == 'analyze_resource.py' else ('True' if rel == 'primary-speed/analyze.py' else 'bool(expected)')
        new_path = "source/('runtime/validated_ws10/metal_serving.py'if " + condition + " and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')"
    assert text.count(old_path) == (2 if rel == 'confirmation/analyze.py' else 1), rel
    text = text.replace(old_path, new_path)
    p = w / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    before = ast.dump(Normalize().visit(ast.parse((original / rel).read_text())), include_attributes=False)
    inverse_paths = text.replace(new_path, old_path)
    after = ast.dump(Normalize().visit(ast.parse(inverse_paths.replace('worldsense-validated-text-worker', 'worldsense-isolated-primary'))), include_attributes=False)
    assert before == after, rel
    checks[rel] = True
for name in ['freeze.py', 'dependency_closure.py', 'controller.py', 'prepare_cohort.py', 'run_resource.py', 'audit_native_cohort.py']:
    text = (seed / name).read_text().replace('worldsense-native-text-worker', 'worldsense-validated-text-worker')
    if name == 'freeze.py':
        needle = "'GARNET_METAL_NATIVE_TEXT_WORKER':('OFF','ON'),"
        assert text.count(needle) == 1
        text = text.replace(needle, needle + "'GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW':('OFF','ON'),")
        needle = "'GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_NATIVE_TEXT_WORKER']"
        assert text.count(needle) == 1
        text = text.replace(needle, "'GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_NATIVE_TEXT_WORKER','GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW']")
        needle = "(w/'environment.json').write_text"
        additions = """old_source=root/'out/research/worldsense-prefix-cache/source'
legacy=json.loads((w/'validated-source-manifest.json').read_text())
assert len(legacy)==1706 and all(sha(old_source/k)==v for k,v in legacy.items())
old_library=root/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'
assert sha(old_library)=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
env['legacy_backend']={'library':str(old_library),'sha256':sha(old_library),'source':str(old_source),'source_manifest_sha256':sha(w/'validated-source-manifest.json')}
env['native_hosts']={role:sha(root/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib')for role in ['on','primary','fault']}
"""
        assert text.count(needle) == 1
        text = text.replace(needle, additions + needle)
        text = text.replace('WS58', 'WS61').replace('rawcandidate10flags', 'rawcandidate11flags')
        text = text.replace('Same-source RAW child', 'Pinned original WS10 RAW child with separate native instrumentation')
        text = text.replace('10flagcontrast,4baseflags', '11flagcontrast,5baseflags')
    elif name == 'dependency_closure.py':
        needle = "    for mode in ['off','primary','on','fault']:\n        b=r/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode)"
        assert text.count(needle) == 1
        text = text.replace(needle, "    for b in [r/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode)for mode in ['off','primary','on','fault']]+[r/'out/build/garnet-metal-worldsense-prefix-cache']:")
        needle = '    assert len(raw)>=30\n'
        additions = """    legacy=r/'out/research/worldsense-prefix-cache/source'
    paths.update(p for p in legacy.rglob('*')if p.is_file()and '__pycache__'not in p.parts and '.git'not in p.parts)
    paths.add(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')
    paths.update(r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib'for role in ['on','primary','fault'])
"""
        assert text.count(needle) == 1
        text = text.replace(needle, needle + additions)
        text = text.replace('Union of compiler-generated dependency files for four builds', 'Union of compiler-generated dependency files for four parent/native-host builds and pinned old build, complete old source, exact old DLL and native-host DLLs')
    elif name == 'prepare_cohort.py':
        text = text.replace("subprocess.run(['python3',str(w/'freeze.py')],check=True)", "assert json.loads((w/'functional-smoke/functional-verification.json').read_text())['four_calls_exact']\nassert json.loads((w/'full-proof-complete.json').read_text())['parent_legacy_model_runtime_assets_and_frozen_drivers_unchanged']\nsubprocess.run(['python3',str(w/'freeze.py')],check=True)")
        text = text.replace("['nine_expected_config_rejections']", "['thirteen_expected_config_rejections']")
        text = text.replace("int(json.loads((w/'reproof-observed-terminal.json').read_text())['session_id'])", "int(json.loads((w/'learned-independent-audit.json').read_text())['controller_session'])")
        text = text.replace("assert json.loads((w/'reproof-complete.json').read_text())['full_native_text_VLM_corpus_vision_lifecycle_and_parent_telemetry_reproof_pass']", "assert json.loads((w/'learned-independent-audit.json').read_text())['original_learned_native_state_token_vision_verifier_pass']")
    elif name == 'audit_native_cohort.py':
        text = text.replace('Extra WS58', 'Extra WS61')
        needle = "(w/'cohort-native-identity-audit.json').write_text"
        additions = """assert hashlib.sha256((r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib').read_bytes()).hexdigest()==environment['legacy_backend']['sha256']
for role,value in environment['native_hosts'].items():
 assert hashlib.sha256((r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib').read_bytes()).hexdigest()==value
"""
        assert text.count(needle) == 1
        text = text.replace(needle, additions + needle)
    ast.parse(text)
    (w / name).write_text(text)
audit = (opt / 'audit_routes_ws58.py').read_text().replace('worldsense-native-text-worker', 'worldsense-validated-text-worker')
needle = "   if c['model_id']=='Qwen3-1.7B':\n"
extra = """   if mode!='off':
    mutations=[(['native_gateway','parent_backend'],'wrong-parent')]
    if c['timings']['native_gateway']['child']is not None:mutations += [(['native_gateway','child','model_backend','library_path'],'/wrong/libgarnet.dylib'),(['native_gateway','child','model_backend','library_sha256'],'wrong'),(['native_gateway','child','model_backend','loaded_garnet_images'],2),(['native_gateway','child','model_backend','package_abi'],0),(['native_gateway','child','model_backend','expected_original_source_root'],'/wrong/source'),(['native_gateway','child','decoder_source'],'/wrong/decoder.py'),(['native_gateway','child','serving_source'],'/wrong/metal_serving.py'),(['native_gateway','child','raw_counter_policy'],'wrong'),(['native_gateway','child','raw_counter_origin'],'wrong')]
    if c['model_id']=='Qwen3-1.7B':mutations += [(['counter_origin'],'wrong'),(['serving_source'],'/wrong/metal_serving.py'),(['lazy_default_retired_bytes'],False)]
    for path,value in mutations:
     bad=copy.deepcopy(c);target=bad['timings']
     for key in path[:-1]:target=target[key]
     target[path[-1]]=value;reject(bad)
"""
assert audit.count(needle) == 1
audit = audit.replace(needle, extra + needle)
ast.parse(audit)
(opt / 'audit_routes_ws61.py').write_text(audit)
(w / 'acceptance-gate-audit.json').write_text(json.dumps(dict(
    all_five_original_golden_numeric_metric_gate_AST_unchanged=True, files=checks,
    only_native_architecture_admission_replaced=True, full_route_mutation_audit_pending=True,
    exact_model_role_serving_path_assertions_inverse_normalized=True,
    driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in rels}), indent=2) + '\n')
print('WS61 five original numerical/metric/golden threshold ASTs unchanged; dual backend freeze/dependency drivers prepared')
