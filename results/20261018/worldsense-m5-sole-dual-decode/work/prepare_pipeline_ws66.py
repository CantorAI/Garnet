from pathlib import Path
import ast
b=Path(__file__).resolve().parent
# Original21 exclusion cases plus the two dependency closures for this distinct flag.
t=(b/'admission_ws65.py').read_text().replace('worldsense-sole-interleaved-text','worldsense-sole-dual-decode').replace('GARNET_METAL_SOLE_INTERLEAVED_TEXT','GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE').replace('Sole interleaved text requires sole MLP storage and direct prefill','Sole dual decode requires sole MLP storage and direct prefill').replace("'text-direct'","'dual-direct'").replace("'text-storage'","'dual-storage'");ast.parse(t);(b/'admission_ws66.py').write_text(t)
t=(b/'prepare_worldsense_ws64.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-dual-decode').replace("['twentyone_expected_config_rejections']","['twentythree_expected_config_rejections']")
t=t.replace('assert t.count(needle)==2','assert t.count(needle)==3')
needle=" if ('sole_direct_interleaved_prefill'in call['backend'])!=enabled:return False"
extra='''
 if ('sole_dual_projection_decode'in call['backend'])!=enabled:return False
 fields_dual={'encoded_pairs','represented_projections'};dual_groups=[t.get(key)for key in ['dual_projection_before','dual_projection_after','dual_projection_delta']]
 if any(not isinstance(group,dict)or set(group)!=fields_dual for group in dual_groups):return False
 if any(not isinstance(value,int)or isinstance(value,bool)or value<0 for group in dual_groups for value in group.values()):return False
 db,da,dd=dual_groups
 if any(dd[k]!=da[k]-db[k]for k in dd)or any(group['represented_projections']!=2*group['encoded_pairs']for group in dual_groups):return False
 if not enabled:
  if any(value for group in dual_groups for value in group.values()):return False
 elif call['model_id']=='Qwen3-1.7B':
  if dd['encoded_pairs']<=0 or dd['encoded_pairs']<28*(call['generated_tokens']-1):return False
 elif any(dd.values()):return False
''';assert t.count(needle)==1;t=t.replace(needle,needle+extra)
t=t.replace("['sole_interleaved_mlp','sole_direct_interleaved_prefill','native_text_worker']","['sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','native_text_worker']").replace("['sole_storage_before','sole_storage_after','sole_encoded_delta','native_gateway']","['sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta','native_gateway']")
a="'GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL':('OFF','ON'),";assert t.count(a)==1;t=t.replace(a,a+"'GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE':('OFF','ON'),")
a="'GARNET_METAL_SOLE_INTERLEAVED_MLP','GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL']}";assert t.count(a)==1;t=t.replace(a,"'GARNET_METAL_SOLE_INTERLEAVED_MLP','GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL','GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE']}")
a="p=w/'freeze.py';t=p.read_text();";assert t.count(a)==1
z="""p=w/'dependency_closure.py';v=p.read_text().replace("['off','primary','on','fault','restore']","['off','primary','on','fault','restore','direct64']").replace('for four builds','for six builds')
needle='    compiler=Path(subprocess.check_output'
extra='''    pair_cmd=list(cmd);pair_cmd[-1]=str(w/'dual_pair_corpus_test.cpp');pair_text=subprocess.check_output(pair_cmd,text=True)
    (w/'paired-corpus-preprocessor-dependencies.txt').write_text(pair_text);paths.update(Path(name)for name in dependencies(pair_text))
    paths.update([w/'dual_pair_corpus_test.cpp',r/'out/build/garnet-metal-worldsense-sole-dual-decode-on/dual_pair_corpus_test'])
'''
assert v.count(needle)==1;v=v.replace(needle,extra+needle);v=v.replace("'preprocessor_command':cmd,","'preprocessor_command':cmd,'paired_preprocessor_command':pair_cmd,");ast.parse(v);p.write_text(v)
p=w/'freeze.py';t=p.read_text();t=t.replace('WS54 fully-isolated wideVLM/rawtext partition; rawcandidate9flags andstoragecontrolcandidate6. Sameprimarypartition/globalretirement, differentmetadata/prefill/directupload. Faults onlyinseparatecorrectnessbuild','WS66 sole paired text decode with unchanged separate outputs/SiLU. RAWcandidate12flags/PRIMARYcandidate9flags. New guarded graph/context inverse exact and all original shader strings/tests unchanged. Faults only separate correctness build');"""
t=t.replace(a,z).replace('WS64 source/elevenflagRAWcontrast/eightflagPRIMARYcontrast','WS66 source/twelveflagRAWcontrast/nineflagPRIMARYcontrast');ast.parse(t);(b/'prepare_worldsense_ws66.py').write_text(t)
t=(b/'audit_routes_ws64.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-dual-decode').replace("'sole_interleaved_mlp','sole_direct_interleaved_prefill']","'sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode']").replace("['sole_storage_before','sole_storage_after','sole_encoded_delta']","['sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta']")
a="   bad=copy.deepcopy(c);bad['model_id']='unknown';reject(bad)";assert t.count(a)==1;t=t.replace(a,"   for field in ['encoded_pairs','represented_projections']:\n    bad=copy.deepcopy(c);bad['timings']['dual_projection_delta'][field]+=1;reject(bad)\n"+a)
t=t.replace("['sole_storage_before','sole_storage_after','sole_encoded_delta','native_gateway']","['sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta','native_gateway']");ast.parse(t);(b/'audit_routes_ws66.py').write_text(t)
for name in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed']:
 t=(b/(name+'_ws64.py')).read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-dual-decode').replace('ws64.py','ws66.py').replace('WS64','WS66').replace('==1721','==1723').replace("['off','primary','on','fault','restore']","['off','primary','on','fault','restore','direct64']").replace('all_five_builds_FP32_words_identical','all_six_builds_FP32_words_identical').replace('twentyone_expected_config_rejections','twentythree_expected_config_rejections')
 if name=='prepare_resource':
  t=t.replace("['original_shader_literals_and_63_decode_graph_session_xModel_image_CUDA_original_tests_unchanged']","['original_shader_literals_and_all_original_eight_tests_unchanged']")
  needle="assert json.loads((w/'learned-independent-audit.json').read_text())['three_actual_direct_prefill_tile_readers_each_FP32_words']==1056964608";assert t.count(needle)==1;t=t.replace(needle,needle+"\nassert json.loads((w/'learned-independent-audit.json').read_text())['all56_actual_gate_up_paired_vector_helper_original_FP32_words']==704643072\nassert json.loads((w/'learned-independent-audit.json').read_text())['actual_generic_resident_shared_rebuild_routes_and_profile_fallback_exact']\nassert json.loads((w/'dual-word-comparison.json').read_text())['all_six_builds_complete_FP32_words_identical']\nassert json.loads((w/'source-inverse-proof.json').read_text())['all_existing_graph_context_bytes_inverse_exact_outside_new_guarded_blocks']")
 ast.parse(t);(b/(name+'_ws66.py')).write_text(t)
print('WS66 future original-gate pipeline preparation/admission helpers generated; no dispatch')
