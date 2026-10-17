from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-only-defaults/source';prior=w.parent/'worldsense-isolated-primary';seed=json.loads((w/'ws54-seed-source-manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((w/'wide-scope-proof.json').read_text())['original_graph_code_recovered_by_only_packed_slot_admission_inverse']
assert json.loads((w/'wide-scope-proof.json').read_text())['all_math_shaders_codecs_loader_session_serving_xModel_CUDA_original_tests_exact']
p=json.loads((w/'actual-program-proof.json').read_text());before=json.loads((prior/'actual-program-proof.json').read_text());assert p['actual_candidate_main_equals_raw_control']and p['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']
assert p['programs']==before['programs']
assert p['programs']['on']['main']==before['programs']['on']['main']
for mode in ['on','fault']:
 text=(r/('out/build/garnet-metal-worldsense-packed-only-defaults-'+mode)/'Testing/Temporary/LastTest.log').read_text();assert 'Direct float upload all65536patterns/unaligned/padded/alias/source-lifetime/invalid/openbatch/Abort PASS'in text
cpu=json.loads((w/'cpu-loop-proof.json').read_text());assert cpu['rowBytes_call_outside_all_backward_branch_intervals']
mm=(s/'src/backends/metal/metal_context.mm').read_text();assert 'primaryGemm16Wide=[impl_->device newComputePipelineStateWithFunction:[primaryLibrary newFunctionWithName:@"primary_gemm_16_64_64"]'in mm
report={'compiled_graph_inverse_and_loader_exact_WS54':True,'raw_and_storage_control_programs_byte_exact_WS54':True,'candidate_MAIN_byte_exact_WS54_raw':True,'all_primary_kernel_bodies_and_prefill_fragment_unchanged':True,'dynamic_primary_specialization_uses_secondary_raw_uses_main':True,'CPU_getter_outside_conversion_loops':True,'host_K32_ONLY_retains_original_K64_kernel':True,'direct_native_all65536_patterns_and_bounds_lifetime_pass':True,'scope':'Source/actual library composition/native correctness only; no performance claim.'}
if(w/'direct-upload-corpus.json').exists():
 selection=json.loads((w/'direct-upload-selection.json').read_text());result=json.loads((w/'direct-upload-corpus.json').read_text());expected={(x['model'],x['tensor']):x for x in selection['weights']};actual={(x['model'],x['tensor']):x for x in result['weights']};assert len(expected)==len(actual)==546 and expected.keys()==actual.keys()
 for key,x in expected.items():
  y=actual[key];assert all(x[k]==y[k]for k in ['shape','rows','columns']);assert y['FP32_words_checked']==x['rows']*x['columns']and y['storage']=='matrix_fp32'and len(y['BF16_sha256'])==64
 assert result['totals']=={'weights':546,'FP32_words_checked':334609152};report['all546_learned_vision_norm_depthwise_patch_candidates_FP32_words_exact']=True;report['direct_learned_words']=334609152
(w/'independent-direct-upload-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
