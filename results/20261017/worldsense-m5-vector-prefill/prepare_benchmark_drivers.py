from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-primary-u32';newer=w.parent/'worldsense-wide-primary-lazy'
files=['freeze.py','prepare_cohort.py','run_resource.py','analyze_resource.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']
for rel in files:
 p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((seed/rel).read_text().replace('worldsense-primary-u32','worldsense-vector-prefill'))
route='''def native_route(call,enabled,primary_control=False):
 packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata']:
  if (cap in call['backend'])!=packed:return False
 if ('primary_vector_prefill'in call['backend'])!=enabled:return False
 if any(cap in call['backend']for cap in ['lazy_resident_defaults','wide_primary_storage','wide_checkpoint_storage']):return False
 if t['lazy_default_retired_bytes']!=0 or t['lazy_default_rehydrated_bytes']!=0 or t['lazy_default_planned_slots']!=0:return False
 if t['memory_after_decode']['task_info_status']!=0:return False
 if not packed:return all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 expected={197:(3440902144,2623650048,13441024),201:(8401715200,6402163712,32819200),398:(11842617344,9025813760,46260224)}
 if s['active_weights']not in expected:return False
 if call['model_id']=='Qwen3.5-4B':
  if s['active_weights']not in [201,398]:return False
 elif call['model_id']=='Qwen3-1.7B':
  if s['active_weights']not in [197,398]:return False
 else:return False
 original,retained,meta=expected[s['active_weights']]
 return s['original_bytes']==original and s['payload_bytes']+s['metadata_bytes']==retained and s['metadata_bytes']==meta and 0<s['allocated_bytes']<original and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']'''
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
 p=w/rel;t=p.read_text();fn=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');old=ast.get_source_segment(t,fn);assert t.count(old)==1;p.write_text(t.replace(old,route,1))
p=w/'freeze.py';t=p.read_text().replace("for mode in ['off','on']:","for mode in ['off','primary','on']:")
a="=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON')}";assert t.count(a)==1;t=t.replace(a,"=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON')}",1)
a="assert all(flags[m]['GARNET_METAL_LAZY_RESIDENT_DEFAULTS']=='OFF'and flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']=='OFF'for m in ['off','on'])";assert t.count(a)==1
t=t.replace(a,a+"\nassert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON')}",1)
a="(w/'starting-source-manifest.json').write_text";assert t.count(a)==1;t=t.replace(a,"assert manifest==json.loads((w/'correctness-source-manifest.json').read_text()),'Source changed since correctness'\n"+a,1)
t=t.replace('only default-OFF primaryBF16 storage flag differs','only primary and vector-prefill flags differ in raw/candidate; prior-primary differs from candidate only by vector-prefill; lazy and cached matrix metadata OFF')
p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text();start=t.index("assert all(v==0");end=t.index("assert all(v['numeric_golden_source_metric_AST_unchanged']",start)
t=t[:start]+'''assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert json.loads((w/'correctness-terminal.json').read_text())['all_three_actual_model_modes_and_five_readers_exact']
assert json.loads((w/'native-word-comparison.json').read_text())['all_four_builds_FP32_words_identical']
'''+t[end:]
t=t.replace("for mode in ['off','on','fault']:","for mode in ['off','primary','on','fault']:")
t=t.replace("paths=[p for p in w.rglob('*.py')if '__pycache__'not in p.parts]","paths=[p for p in w.rglob('*')if p.is_file()and p.suffix in ['.py','.cmake']and '__pycache__'not in p.parts]")
p.write_text(t)
p=w/'dependency_closure.py';t=p.read_text().replace("for mode in ['off','on','fault']:","for mode in ['off','primary','on','fault']:").replace('for three builds','for four builds')
a="    assert len(raw)>=30";assert t.count(a)==1;t=t.replace(a,a+"\n    paths.update(p for p in (w/'build-profile').glob('*.cmake')if p.is_file())",1);p.write_text(t)
# Original full confirmation now requires every preceding screen.
p=w/'confirmation/prepare.py';t=p.read_text();a="assert json.loads((p/'speed/analysis.json').read_text())['passed']";assert t.count(a)==1;t=t.replace(a,a+"\nassert json.loads((p/'primary-speed/analysis.json').read_text())['passed']\nassert json.loads((p/'production-speed/analysis.json').read_text())['passed']",1);p.write_text(t)
p=w/'speed/prepare.py';t=p.read_text();a="assert json.loads((p/'resource-analysis.json').read_text())['passed']";assert t.count(a)==1;t=t.replace(a,a+"\nassert json.loads((p/'primary-speed/analysis.json').read_text())['passed']",1);p.write_text(t)
# Additional actual deployed WS10 screen, existing budgets preserved.
d=w/'production-speed';d.mkdir()
for name in ['prepare.py','run_stage.py','analyze.py']:
 t=(newer/'production-speed'/name).read_text().replace('worldsense-wide-primary-lazy','worldsense-vector-prefill');(d/name).write_text(t)
p=d/'analyze.py';t=p.read_text();fn=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');old=ast.get_source_segment(t,fn)
guard=" if not enabled:\n  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_vector_prefill','lazy_resident_defaults'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])\n"
actual_route=route.replace(" packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']",guard+" packed=enabled or primary_control;t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']",1)
p.write_text(t.replace(old,actual_route,1))
# Matched primary-only control screen; resource gate remains a separate prerequisite.
d=w/'primary-speed';d.mkdir()
t=(w/'speed/prepare.py').read_text();a="assert json.loads((p/'primary-speed/analysis.json').read_text())['passed']\n";assert t.count(a)==1;t=t.replace(a,'',1)
a="for name in ['environment.json','starting-source-manifest.json']:shutil.copy2(p/name,w/name)";assert t.count(a)==1
t=t.replace(a,a+"\nenv=json.loads((w/'environment.json').read_text());env['libraries']['off']=env['libraries']['primary'];env['comparison_scope']='Only PRIMARY_VECTOR_PREFILL differs; both full-primary and eager';(w/'environment.json').write_text(json.dumps(env,indent=2))",1)
t=t.replace(".replace(\"stage(w/'before-libgarnet.dylib')\"", ".replace(\"else'off'))/'libgarnet.dylib'\",\"else'primary'))/'libgarnet.dylib'\").replace(\"stage(w/'before-libgarnet.dylib')\"",1)
t=t.replace('worldsense-vector-prefill/speed','worldsense-vector-prefill/primary-speed')
(d/'prepare.py').write_text(t)
t=(w/'speed/run_stage.py').read_text().replace("'speed'","'primary-speed'");(d/'run_stage.py').write_text(t)
t=(w/'speed/analyze.py').read_text().replace('native=native_route(c,bool(expected))','native=native_route(c,bool(expected),True)')
a='passed=all(';start=t.index(a);end=t.index('\nresult=dict(',start)
t=t[:start]+'''foot={label:max(r['call']['timings']['memory_after_decode']['task_physical_footprint_bytes']for r in rr)for label,rr in rows.items()}
matched_memory_preserved=mem['candidate-1']/mem['control-1']-1<=.02 and foot['candidate-1']/foot['control-1']-1<=.02
vlm_ttft_gain=1-metrics['candidate-1']['Qwen3.5-4B']['ttft']/metrics['control-1']['Qwen3.5-4B']['ttft']
passed=all(c['exact']and c['native_lazy_defaults_valid']for c in checks)and matched_memory_preserved and vlm_ttft_gain>=.05 and wall_gain>=-.02 and all(c['ttft_change']<=.02 and c['total_change']<=.02and c['decode_gain']>=-.02 for c in comparisons)'''+t[end:]
t=t.replace("result=dict(status=","result=dict(matched_primary=True,matched_memory_preserved=matched_memory_preserved,sampled_max_footprint_bytes=foot,minimum_VLM_TTFT_gain=.05,observed_VLM_TTFT_gain=vlm_ttft_gain,status=",1)
t=t.replace('Fresh single-car screen.','Matched primary-only/vector-prefill screen, exact common storage/lifetimes;5%VLMwarmTTFT improvement and2%model/wall/memory preservation required. Not isolated GPU timing or confirmed causal evidence.')
(d/'analyze.py').write_text(t)
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'benchmark-preparation-scope.json').write_text(json.dumps({'no_worker_dispatched':True,'new_matched_primary20_before_rawspeed20':True,'unchanged_original_quantitative_and_golden_AST_to_be_audited':True,'new5percentVLM_TTFT_and2percent_memory_preservation_gate_declared_in_protocol':True,'profile_two_CMake_files_in_dependencies_and_driver_identities':True},indent=2)+'\n')
print('WS47 resource8/matched-primary20/raw20/actual20/full48 prospective drivers prepared; no benchmark dispatched.')
