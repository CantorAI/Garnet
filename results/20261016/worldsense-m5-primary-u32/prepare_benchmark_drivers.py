"""Prepare prospective drivers; no benchmark worker is dispatched here."""
from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-primary-kmajor'
for rel in ['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']:
    p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text((old/rel).read_text().replace('worldsense-primary-kmajor','worldsense-primary-u32'))
old_expected='197:(3440902144,2637091072,26882048),201:(8401715200,6434982912,65638400),398:(11842617344,9072073984,92520448)'
new_expected='197:(3440902144,2623650048,13441024),201:(8401715200,6402163712,32819200),398:(11842617344,9025813760,46260224)'
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
    p=w/rel;t=p.read_text();assert old_expected in t;t=t.replace(old_expected,new_expected,1)
    needle=" if ('primary_kmajor_decode'in call['backend'])!=enabled:return False";assert needle in t
    p.write_text(t.replace(needle,needle+"\n if ('primary_u32_metadata'in call['backend'])!=enabled:return False",1))
p=w/'audit_gates.py';t=p.read_text();t=t.replace("old=w.parent/'worldsense-primary-vector'","old=w.parent/'worldsense-primary-kmajor'")
t=t.replace(".replace('worldsense-primary-u32','worldsense-primary-vector')",".replace('worldsense-primary-u32','worldsense-primary-kmajor')")
needle="   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_kmajor_decode','')if mode=='on'else bad['backend']+' primary_kmajor_decode';assert not namespace['native_route'](bad,mode=='on')";assert needle in t
t=t.replace(needle,needle+"\n   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_u32_metadata','')if mode=='on'else bad['backend']+' primary_u32_metadata';assert not namespace['native_route'](bad,mode=='on')\n   bad=copy.deepcopy(c);bad['timings']['primary_storage_after']['metadata_bytes']+=4;assert not namespace['native_route'](bad,mode=='on')",1)
t=t.replace("'wrong_primary_vector_Kmajor_capability_rehydration_or_vector_route_counts_rejected'","'wrong_primary_vector_Kmajor_u32_capability_metadata_width_rehydration_or_route_counts_rejected'")
p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text();needle="assert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_Kmajor_GPU_bit_reconstruction']";assert needle in t
p.write_text(t.replace(needle,needle+"\nassert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_u32_GPU_bit_reconstruction']",1))
(w/'prospective-storage-counters.json').write_text(json.dumps({'scope':'Before timing, from unchanged WS39 payload sums and mathematically halved metadata; validated with actual native correctness calls before any benchmark.','large_matrices_197':{'original':3440902144,'payload_plus_meta':2623650048,'meta':13441024},'large_matrices_201':{'original':8401715200,'payload_plus_meta':6402163712,'meta':32819200},'large_matrices_398':{'original':11842617344,'payload_plus_meta':9025813760,'meta':46260224},'primary_plus_small_raw':9033678080},indent=2))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('Future8/20/48 drivers prepared with prospective32bit storage counters; no benchmark dispatched',flush=True)
