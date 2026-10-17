"""Prepare future benchmark drivers only; dispatch is separately gated on correctness."""
from pathlib import Path
import ast, hashlib, json

w=Path(__file__).resolve().parent
old=w.parent/'worldsense-primary-gpu'
for rel in ['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py',
            'analyze_resource.py','audit_gates.py','speed/prepare.py','speed/run_stage.py',
            'speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']:
    p=w/rel
    assert not p.exists(),rel
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text((old/rel).read_text().replace('worldsense-primary-gpu','worldsense-primary-vector'))

route='''def native_route(call,enabled):
 t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 if ('primary_bf16_blocks'in call['backend'])!=enabled:return False
 if ('primary_vector_decode'in call['backend'])!=enabled:return False
 if t['lazy_default_retired_bytes']!=0 or t['lazy_default_rehydrated_bytes']!=0:return False
 if t['memory_after_decode']['task_info_status']!=0:return False
 if not enabled:return all(v==0 for v in s.values())and all(v==0 for v in delta.values())
 expected={197:(3440902144,2637091072,26882048),201:(8401715200,6434982912,65638400),398:(11842617344,9072073984,92520448)}
 if s['active_weights']not in expected:return False
 original,retained,meta=expected[s['active_weights']]
 return s['original_bytes']==original and s['payload_bytes']+s['metadata_bytes']==retained and s['metadata_bytes']==meta and 0<s['allocated_bytes']<original and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']) and delta['vector_decode_encoded']==delta['decode_encoded'] and delta['vector_head_encoded']==delta['head_encoded']
'''
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
    p=w/rel;t=p.read_text();tree=ast.parse(t);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_route')
    lines=t.splitlines(True);lines[fn.lineno-1:fn.end_lineno]=[route];p.write_text(''.join(lines))
p=w/'audit_gates.py';t=p.read_text()
t=t.replace("old=w.parent/'worldsense-cached-matrix-metadata'","old=w.parent/'worldsense-primary-gpu'")
t=t.replace(".replace('worldsense-primary-vector','worldsense-cached-matrix-metadata')",".replace('worldsense-primary-vector','worldsense-primary-gpu')")
needle="   bad=copy.deepcopy(c);bad['timings']['lazy_default_rehydrated_bytes']=1;assert not namespace['native_route'](bad,mode=='on')"
assert needle in t
t=t.replace(needle,needle+'''
   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_vector_decode','')if mode=='on'else bad['backend']+' primary_vector_decode';assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['vector_decode_encoded']+=1;assert not namespace['native_route'](bad,mode=='on')
   bad=copy.deepcopy(c);bad['timings']['primary_encoded_delta']['vector_head_encoded']+=1;assert not namespace['native_route'](bad,mode=='on')
''',1)
t=t.replace("'wrong_metadata_capability_and_rehydration_rejected':True","'wrong_primary_or_vector_capability_rehydration_or_vector_route_counts_rejected':True")
p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text();needle="assert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_GPU_bit_reconstruction']";assert needle in t
t=t.replace(needle,needle+"\nassert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_vector_GPU_bit_reconstruction']",1);p.write_text(t)

# Explicit common-seed identity, checked without mutating any consumed source.
source=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-gpu/source')
manifest=json.loads((old/'starting-source-manifest.json').read_text())
assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==digest for rel,digest in manifest.items())
(w/'seed-source-verification.json').write_text(json.dumps({'all_WS37_source_files_still_match_frozen_manifest':True,'count':len(manifest),'manifest_archive':'Garnet/results/20261016/worldsense-m5-primary-gpu/starting-source-manifest.json'},indent=2))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('Future resource/speed/full drivers prepared; no benchmark dispatched. Seed source identity PASS',flush=True)
