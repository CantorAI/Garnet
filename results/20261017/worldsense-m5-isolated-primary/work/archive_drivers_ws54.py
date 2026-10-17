from pathlib import Path
import ast
p=Path('work/worldsense-opt');a=(p/'archive_wide_cached_prefill.py').read_text().replace('WS53','WS54').replace('WS-M5-53','WS-M5-54').replace('worldsense-wide-cached-prefill','worldsense-isolated-primary').replace('worldsense-m5-wide-cached-prefill','worldsense-m5-isolated-primary').replace('archive_wide_cached_prefill','archive_isolated_primary').replace('verify_wide_cached_prefill','verify_isolated_primary')
a=a.replace("'seed_ws53.py','merge_ws53.py','drivers_ws53.py','benchmark_ws53.py','finalize_ws53.py','archive_drivers_ws53.py'", "'design_ws54.py','source_ws54.py','drivers_ws54.py','finalize_ws54.py','archive_drivers_ws54.py'")
(p/'archive_isolated_primary.py').write_text(a)
v=(p/'verify_wide_cached_prefill.py').read_text().replace('WS53','WS54').replace('WS-M5-53','WS-M5-54').replace('worldsense-wide-cached-prefill','worldsense-isolated-primary').replace('worldsense-m5-wide-cached-prefill','worldsense-m5-isolated-primary')
v=v.replace('==9 and len(closure)==1717','==10 and len(closure)==1718').replace('source_overlay=9','source_overlay=10').replace("'actual_legacy_main_priorprimary_candidate_fault_byte_exact'","'actual_candidate_main_equals_raw_control'").replace("'actual_candidate_prefill_equals_minimal_preamble_plus_fragment'","'actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment'")
v=v.replace("main=(d/'actual-primary-main-program.metal').read_bytes()", "main=(d/'actual-off-main-program.metal').read_bytes()")
v=v.replace("preamble=(d/'source/src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R\"PMIN(',1)[1].split(')PMIN\"',1)[0]", "preamble=(d/'source/src/backends/metal/primary_isolation_shader.h').read_text().split('R\"IPRIMARY(',1)[1].split(')IPRIMARY\"',1)[0]")
v=v.replace("assert preamble=='#include <metal_stdlib>\\nusing namespace metal;\\n'+abi+'\\n'+rawheader[hstart:hend]", "bodyheader=subprocess.check_output(['git','-C',str(r),'show',base+':src/backends/metal/body_shape_shader.h']).decode();constants=bodyheader.split('R\"BODYSHAPE(',1)[1].split('kernel void ',1)[0]\nassert preamble=='#include <metal_stdlib>\\nusing namespace metal;\\n'+abi+'\\n'+constants")
v=v.replace('preamble.encode()+new.encode()', 'preamble.encode()+original.encode()+new.encode()').replace("b'kernel void primary_bf16_decode'not in(d/'actual-on-prefill-program.metal')", "b'kernel void primary_bf16_decode'in(d/'actual-on-prefill-program.metal')")
v=v.replace('[library newFunctionWithName:@"primary_bf16_decode"]','[primaryLibrary newFunctionWithName:@"primary_bf16_decode"]').replace('[library newFunctionWithName:@"primary_gemm_16_64_64"]','[primaryLibrary newFunctionWithName:@"primary_gemm_16_64_64"]')
v=v.replace("['entire_compiled_loader_exact_reconstructed_merge']", "['full_context_recovered_by_only_library_composition_and_lookup_inverse']").replace("['actual_four_build_programs_byte_exact_WS50']", "['raw_and_storage_control_programs_byte_exact_WS53']")
a=v.index('# Independently reconstruct wide changes');b=v.index("for mode in ['on','fault']:",a)
# Recover the exact source inventory against immutable prior Git artifacts,
# then independently enforce the declared four-file delta and context inverse.
inverse=(Path('work/worldsense-isolated-primary/verify_scope.py')).read_text();x=inverse.index('def replace(');y=inverse.index('header=',x);inverse=inverse[x:y]
block='''# Reconstruct immutable53 context, then invert ONLY the declared54 host delta.
seed53=load(d/'ws53-seed-source-manifest.json');ref53='aebaf00eace8fc4e512db5ca1704ae77af540123'
recon53=json.loads(subprocess.check_output(['git','-C',str(r),'show',ref53+':results/20261017/worldsense-m5-wide-cached-prefill/source-reconstruction.json']))
def bytes53(rel):
 key=('results/20261017/worldsense-m5-wide-cached-prefill/source/'+rel)if rel in recon53['overlay_files']else rel
 data=subprocess.check_output(['git','-C',str(r),'show',(ref53 if rel in recon53['overlay_files']else base)+':'+key]);assert hashlib.sha256(data).hexdigest()==seed53[rel];return data
mm=(d/'source/src/backends/metal/metal_context.mm').read_text();original_context=bytes53('src/backends/metal/metal_context.mm').decode()
'''+inverse.replace('assert mm==original,','assert mm==original_context,')+'''
assert {k for k,h in closure.items()if seed53.get(k)!=h}=={'cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/primary_isolation_shader.h','src/entry/metal_main.cpp'}
entry=(d/'source/src/entry/metal_main.cpp').read_text();marker='#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\\n        info+="; primary_full_isolated_library";\\n#endif\\n';assert entry.count(marker)==1 and entry.replace(marker,'',1).encode()==bytes53('src/entry/metal_main.cpp')
cmake=(d/'source/cmake/GarnetMetal.cmake').read_bytes();assert cmake.startswith(bytes53('cmake/GarnetMetal.cmake'))and b'Full primary isolation requires primary weights and isolated vector prefill'in cmake
prior_program=json.loads(subprocess.check_output(['git','-C',str(r),'show',ref53+':results/20261017/worldsense-m5-wide-cached-prefill/actual-program-proof.json']))
assert all(program['programs'][mode]==prior_program['programs'][mode]for mode in ['off','primary'])
assert program['programs']['on']['main']==prior_program['programs']['off']['main']
assert len(load(d/'flag-dependency-proof.json'))==6 and all(x['expected_rejection']for x in load(d/'flag-dependency-proof.json').values())
'''
v=v[:a]+block+v[b:]
v=v.replace("assert ('primary_prefill_isolated_library'in c['backend'])==enabled", "assert ('primary_prefill_isolated_library'in c['backend'])==enabled\n                    assert ('primary_full_isolated_library'in c['backend'])==enabled")
v=v.replace("==80721", "==20252").replace("==25263", "==59577")
ast.parse(v);(p/'verify_isolated_primary.py').write_text(v)
print('WS54 archive and independent audit drivers prepared')
