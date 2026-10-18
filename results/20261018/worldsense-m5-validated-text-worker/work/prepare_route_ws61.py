"""Extend the original native route only for explicit dual-backend provenance."""
from pathlib import Path
import ast, hashlib, json
opt = Path(__file__).resolve().parent
seed = (opt / 'ws58_native_route.py').read_text()
text = seed.replace('WS58', 'WS61').replace('worldsense-native-text-worker', 'worldsense-validated-text-worker')
old = "   if support['library_path']!='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-validated-text-worker-off/libgarnet.dylib'or support['source_root']!='/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source':return False"
new = """   role='on'if enabled else'primary'
   if support['library_path']!='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-validated-text-worker-'+role+'/libgarnet_worker_host.dylib'or support['source_root']!='/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source':return False
   backend=child['model_backend']
   if backend['library_path']!='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'or backend['library_sha256']!='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'or backend['loaded_garnet_images']!=1 or backend['package_abi']<=0:return False
   if backend['expected_original_source_root']!='/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source':return False
   if child['decoder_source']!='/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source/xModel/qwen/metal_dense/decoder.py'or child['serving_source']!='/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source/runtime/validated_ws10/metal_serving.py':return False
   if child['raw_counter_policy']!='pinned_ws10_structural_absence'or not isinstance(child['raw_counter_origin'],str)or'Structural zeros'not in child['raw_counter_origin']:return False
   if remote:
    if t['serving_source']!=child['serving_source']or t['counter_origin']!=child['raw_counter_origin']:return False
    for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:
     if any(not isinstance(v,int)or isinstance(v,bool)or v!=0 for v in t[key].values()):return False
    for key in ['lazy_default_planned_slots','lazy_default_retired_bytes','lazy_default_rehydrated_bytes']:
     if not isinstance(t[key],int)or isinstance(t[key],bool)or t[key]!=0:return False
""".rstrip()
assert text.count(old) == 1
text = text.replace(old, new)
needle = "  if g['route']!=('native_lrpc_raw_text'if remote else'local')or'native_text_worker'not in parent:return False\n"
extra = "  if 'native_text_worker_validated_ws10'not in parent:return False\n"
assert text.count(needle) == 1
text = text.replace(needle, needle + extra)
ast.parse(text)
(opt / 'ws61_native_route.py').write_text(text)
inverse = text.replace(new, old).replace(extra, '').replace('WS61', 'WS58').replace('worldsense-validated-text-worker', 'worldsense-native-text-worker')
assert inverse == seed
(opt / 'ws61-route-adaptation-proof.json').write_text(json.dumps(dict(
    original_clock_memory_model_route_ownership_and_counter_checks_inverse_exact=True,
    added_dual_source_library_overlay_and_structural_counter_provenance=True,
    seed_sha256=hashlib.sha256(seed.encode()).hexdigest(),
    current_sha256=hashlib.sha256(text.encode()).hexdigest(),
    scope='Admission only. No numeric, golden, metric or performance threshold changed.'), indent=2) + '\n')
print('WS61 route prepared; original admission retained plus pinned old backend/new native host checks')
