from pathlib import Path
import ast,json
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');prior=root/'work/worldsense-isolated-primary';w=root/'work/worldsense-raw-runtime-diagnostic'
s=(prior/'production-speed/analyze.py').read_text();tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');lines=s.splitlines(keepends=True)
route='''def native_route(call,enabled,primary_control=False):
 forbidden=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','lazy_resident_defaults','wide_primary_storage','wide_checkpoint_storage']
 if 'dense_prefix_cache'not in call['backend']or any(k in call['backend']for k in forbidden):return False
 t=call['timings']
 if not enabled:return not any(k in t for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
 if t['memory_after_decode']['task_info_status']!=0:return False
 if any(t[k]!=0 for k in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']):return False
 return all(v==0 for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta']for v in t[key].values())
'''
s=''.join(lines[:fn.lineno-1])+route+''.join(lines[fn.end_lineno:]);s=s.replace('import json,statistics,hashlib','import json,statistics,hashlib,sys')
s=s.replace("labels=['control-1','candidate-1'];", "labels=sys.argv[1:3];assert len(labels)==2;control,candidate=labels;assert control.startswith('control')and candidate.startswith('candidate');")
s=s.replace("'control-1'",'control').replace("'candidate-1'",'candidate')
s=s.replace("status='passed_screen_requires_fresh_full_confirmation'if passed else'rejected',passed=passed", "status='diagnostic_within_2percent'if passed else'diagnostic_difference',preserved_within_2percent=passed,diagnostic_only=True,promote=False")
s=s.replace("(w/'analysis.json')", "(w/('analysis-'+control+'-'+candidate+'.json'))")
a=s.index("scope='Additional predeclared");b=s.index("')\n(w/",a)
s=s[:a]+"scope='WS55 raw-runtime diagnostic: original warm first/TTFT/total/deriveddecode/Skillwall definitions and2%orientation perpair. No optimization acceptance or promotion. Each pair reported independently; old memory unavailable/null. Existing WS53/54 rejections unchanged."+s[b:]
ast.parse(s);(w/'analyze.py').write_text(s)
env=json.loads((w/'environment.json').read_text());env['control_source']=env['old_source'];(w/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
# Check native admission on real existing rawOFF and oldWS10 calls, plus negatives.
ns={};tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(w/'analyze.py'),'exec'),ns)
records=json.loads((prior/'api-state-progress-off.json').read_text())['records'];assert len(records)==5 and all(ns['native_route'](x['call'],True)for x in records)
old=json.loads((prior/'production-speed/control-1.json').read_text());calls=[c for i in old['items']for c in i['result']['model_response'].values()if isinstance(c,dict)and'model_id'in c];assert len(calls)==10 and all(ns['native_route'](c,False)for c in calls)
import copy
for r in records:
 for field in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
  bad=copy.deepcopy(r['call']);bad['timings'][field]=1;assert not ns['native_route'](bad,True)
 for field in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:
  bad=copy.deepcopy(r['call']);k=next(iter(bad['timings'][field]));bad['timings'][field][k]=1;assert not ns['native_route'](bad,True)
 bad=copy.deepcopy(r['call']);bad['backend']+=' primary_bf16_blocks';assert not ns['native_route'](bad,True)
(w/'route-preflight.json').write_text(json.dumps({'actual_newraw_API_calls':5,'actual_prior_oldWS10_calls':10,'nonzero_primary_lifecycle_counters_and_wrong_capability_rejected':True,'scope':'Schema/native-policy audit only; not new performance evidence.'},indent=2)+'\n')
print('WS55 diagnostic analyzer/native admission prepared and checked on15 actual prior calls')
