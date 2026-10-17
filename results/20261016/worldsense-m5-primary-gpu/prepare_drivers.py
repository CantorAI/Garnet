from pathlib import Path
import ast,shutil
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-cached-matrix-metadata'
for cohort in ['speed','confirmation']:(w/cohort).mkdir()
for name in ['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py']:
 t=(old/name).read_text().replace('worldsense-cached-matrix-metadata','worldsense-primary-gpu')
 if name=='freeze.py':
  a=t.index('assert {k:(flags');b=t.index("(w/'build-flag-comparison.json')",a)
  t=t[:a]+"assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON')}\nassert all(flags[m]['GARNET_METAL_LAZY_RESIDENT_DEFAULTS']=='OFF'and flags[m]['GARNET_METAL_CACHED_MATRIX_METADATA']=='OFF'for m in ['off','on'])\n"+t[b:]
  t=t.replace("only declared lazy-retirement/immutable-matrix-metadata flags differ","only default-OFF primaryBF16 storage flag differs")
 if name=='prepare_cohort.py':
  a=t.index("assert json.loads((w/'api-correctness-result");b=t.index("subprocess.run(['python3',str(w/'freeze.py')",a)
  t=t[:a]+"""assert json.loads((w/'api-correctness-result.json').read_text())['all32VLM28text_state_bits_tokens_prefix_capacity_vision_and_layer_isolation']
assert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_GPU_bit_reconstruction']
assert json.loads((w/'native-word-comparison.json').read_text())['all_FP32_words_identical']
assert all(v['numeric_golden_source_metric_AST_unchanged']for v in json.loads((w/'acceptance-gate-audit.json').read_text()).values())
"""+t[b:]
 (w/name).write_text(t)
for cohort in ['speed','confirmation']:
 for name in ['prepare.py','run_stage.py']:(w/cohort/name).write_text((old/cohort/name).read_text().replace('worldsense-cached-matrix-metadata','worldsense-primary-gpu'))
helper="""def native_route(call,enabled):
 t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 if ('primary_bf16_blocks'in call['backend'])!=enabled:return False
 if t['lazy_default_retired_bytes']!=0 or t['lazy_default_rehydrated_bytes']!=0:return False
 if t['memory_after_decode']['task_info_status']!=0:return False
 if not enabled:return all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 expected={197:(3440902144,2637091072,26882048),201:(8401715200,6434982912,65638400),398:(11842617344,9072073984,92520448)}
 if s['active_weights']not in expected:return False
 original,retained,meta=expected[s['active_weights']]
 return s['original_bytes']==original and s['payload_bytes']+s['metadata_bytes']==retained and s['metadata_bytes']==meta and 0<s['allocated_bytes']<original and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])

"""
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
 t=(old/rel).read_text().replace('worldsense-cached-matrix-metadata','worldsense-primary-gpu');tree=ast.parse(t);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');lines=t.splitlines(True);t=''.join(lines[:fn.lineno-1])+helper+''.join(lines[fn.end_lineno:]);(w/rel).write_text(t)
# Gate/metric/semantic/source checks identical except the declared native storage route.
t=(old/'audit_gates.py').read_text().replace("old=w.parent/'worldsense-request-residency'","old=w.parent/'worldsense-cached-matrix-metadata'").replace("worldsense-cached-matrix-metadata','worldsense-request-residency","worldsense-primary-gpu','worldsense-cached-matrix-metadata")
t=t.replace("bad['backend'].replace('cached_matrix_metadata','')if mode=='on'else bad['backend']+' cached_matrix_metadata'","bad['backend'].replace('primary_bf16_blocks','')if mode=='on'else bad['backend']+' primary_bf16_blocks'")
t=t.replace("'all_metric_numeric_golden_and_source_identity_AST_unchanged'","'numeric_golden_source_metric_AST_unchanged'")
# No old experiment-specific column replacement needed: both use retired_default_bytes.
(w/'audit_gates.py').write_text(t)
for p in w.rglob('*.py'):ast.parse(p.read_text())
print('Prepared resource/speed/two-full-pair gates; no performance worker dispatched')
