"""Prepare prospective drivers without running a benchmark."""
from pathlib import Path
import ast, json
w=Path(__file__).resolve().parent
old=w.parent/'worldsense-primary-vector'
for rel in ['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py','analyze_resource.py','audit_gates.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']:
    p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text((old/rel).read_text().replace('worldsense-primary-vector','worldsense-primary-kmajor'))
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
    p=w/rel;t=p.read_text();needle=" if ('primary_vector_decode'in call['backend'])!=enabled:return False";assert needle in t
    p.write_text(t.replace(needle,needle+"\n if ('primary_kmajor_decode'in call['backend'])!=enabled:return False",1))
p=w/'audit_gates.py';t=p.read_text().replace("old=w.parent/'worldsense-primary-gpu'","old=w.parent/'worldsense-primary-vector'")
t=t.replace(".replace('worldsense-primary-kmajor','worldsense-primary-gpu')",".replace('worldsense-primary-kmajor','worldsense-primary-vector')")
needle="   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_vector_decode','')if mode=='on'else bad['backend']+' primary_vector_decode';assert not namespace['native_route'](bad,mode=='on')";assert needle in t
t=t.replace(needle,needle+"\n   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_kmajor_decode','')if mode=='on'else bad['backend']+' primary_kmajor_decode';assert not namespace['native_route'](bad,mode=='on')",1)
t=t.replace("'wrong_primary_or_vector_capability_rehydration_or_vector_route_counts_rejected'","'wrong_primary_vector_Kmajor_capability_rehydration_or_vector_route_counts_rejected'")
p.write_text(t)
p=w/'full_correctness.py';t=p.read_text();needle="'all398actualpacked_weights_vector_GPU_bit_reconstruction':True";assert needle in t
p.write_text(t.replace(needle,needle+",'all398actualpacked_weights_Kmajor_GPU_bit_reconstruction':True",1))
p=w/'prepare_cohort.py';t=p.read_text();needle="assert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_vector_GPU_bit_reconstruction']";assert needle in t
p.write_text(t.replace(needle,needle+"\nassert json.loads((w/'api-correctness-result.json').read_text())['all398actualpacked_weights_Kmajor_GPU_bit_reconstruction']",1))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('Prospective 8/20/48 drivers parse; no benchmark dispatched',flush=True)
