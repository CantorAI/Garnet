from pathlib import Path
import shutil,ast,json
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');old=root/'work/worldsense-wide-cached-prefill';w=root/'work/worldsense-isolated-primary'
for p in list(old.glob('*.py'))+[p for p in old.glob('*/*.py')if p.parent.name in ['primary-speed','production-speed','speed','confirmation']]:
 if p.name in ['build.py','verify_scope.py','verify_direct_upload.py']:continue
 q=w/p.relative_to(old);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace('worldsense-wide-cached-prefill','worldsense-isolated-primary'))
for name in ['seed-source-manifest.json','ws49-seed-source-manifest.json','ws50-seed-source-manifest.json','ws51-seed-source-manifest.json','ws52-seed-source-manifest.json','before-libgarnet.dylib','cases.json','direct-upload-selection.json']:
 shutil.copy2(old/name,w/name)
for name in ['primary_corpus_test.cpp','direct_float_corpus_test.cpp']:(w/name).write_text((old/name).read_text().replace('worldsense-wide-cached-prefill','worldsense-isolated-primary'))
oldmain='actual_legacy_main_priorprimary_candidate_fault_byte_exact';newmain='actual_candidate_main_equals_raw_control'
oldpre='actual_candidate_prefill_equals_minimal_preamble_plus_fragment';newpre='actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment'
for p in w.rglob('*.py'):
 s=p.read_text().replace(oldmain,newmain).replace(oldpre,newpre)
 if p.name in ['full_correctness.py','verify_correctness_terminal.py','prepare_admission.py']:s=s.replace('1717','1718')
 p.write_text(s)
p=w/'native_controller.py';s=p.read_text().replace("assert programs['primary']['main']==programs['on']['main']==programs['fault']['main']", "assert programs['off']['main']==programs['on']['main']==programs['fault']['main']\n prior=json.loads((w.parent/'worldsense-wide-cached-prefill/actual-program-proof.json').read_text());assert all(programs[mode]['main']==(w.parent/'worldsense-wide-cached-prefill'/('actual-'+mode+'-main-program.metal')).read_bytes()for mode in ['off','primary'])")
s=s.replace("preamble=(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R\"PMIN(',1)[1].split(')PMIN\"',1)[0].encode()", "preamble=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R\"IPRIMARY(',1)[1].split(')IPRIMARY\"',1)[0].encode()\n primary=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R\"PRIMARY(',1)[1].split(')PRIMARY\"',1)[0].encode()")
s=s.replace("programs['on']['prefill']==preamble+fragment", "programs['on']['prefill']==preamble+primary+fragment").replace("b'kernel void primary_bf16_decode'not in programs['on']['prefill']", "b'kernel void primary_bf16_decode'in programs['on']['prefill']\n assert b'kernel void primary_bf16_decode'not in programs['on']['main']")
s=s.replace('ACTUAL_MAIN_PROGRAM_PRIORPRIMARY_CANDIDATE_FAULT_IDENTICAL_AND_PREFILL_ISOLATED','ACTUAL_MAIN_CANDIDATE_EQUALS_RAW_WITH_ALL_PRIMARY_KERNELS_IN_SECONDARY')
p.write_text(s)
# Strong capability checks follow existing candidate policy, not global primary marker.
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 p=w/rel;s=p.read_text().replace("'cached_matrix_metadata']:","'cached_matrix_metadata','primary_full_isolated_library']:");p.write_text(s)
p=w/'api_state_guard.py';s=p.read_text().replace("timings=call['timings'];", "assert ('primary_full_isolated_library'in call['backend'])==vector_expected\n timings=call['timings'];");p.write_text(s)
p=w/'audit_gates.py';s=p.read_text().replace("'cached_matrix_metadata']:","'cached_matrix_metadata','primary_full_isolated_library']:");p.write_text(s)
p=w/'freeze.py';s=p.read_text().replace("'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),", "'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON'),",1).replace("'GARNET_METAL_CACHED_MATRIX_METADATA']} ","'GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']} ")
s=s.replace("'GARNET_METAL_CACHED_MATRIX_METADATA']}\n", "'GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}\n").replace('rawcandidate8flags andstoragecontrolcandidate5','rawcandidate9flags andstoragecontrolcandidate6').replace('WS53 wideVLM/rawtext','WS54 fully-isolated wideVLM/rawtext');p.write_text(s)
p=w/'primary-speed/prepare.py';s=p.read_text().replace("'GARNET_METAL_CACHED_MATRIX_METADATA':('OFF','ON')}", "'GARNET_METAL_CACHED_MATRIX_METADATA':('OFF','ON'),'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON')}").replace('Fiveflags(vector/isolation/directupload/K32/cached)','Sixflags(vector/isolation/directupload/K32/cached/full-isolation)');p.write_text(s)
p=w/'prepare_admission.py';s=p.read_text().replace("reference=json.loads((w.parent/'worldsense-primary-cached-lazy/actual-program-proof.json').read_text());assert program['programs']==reference['programs']", "assert program['actual_candidate_main_equals_raw_control']and program['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']")
s=s.replace("assert native['controller_session']==50101 and native['terminal_exit_code']==0", "assert native['terminal_exit_code']==0 and native['all_four_builds_seven_native_tests_pass']")
s=s.replace("'actual_four_programs_byte_exact_WS52':True", "'raw_MAIN_and_primary_fragments_exact_WS53':True").replace('predeclared53design','predeclared54design').replace('Fourinvalidconfigguards, sourcewideScope/program52identity','Full-isolation config/source/library composition guards')
p.write_text(s)
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('WS54 core drivers cloned/adapted; independent scope/direct-upload proofs and admission additions required before any native/timing dispatch')
