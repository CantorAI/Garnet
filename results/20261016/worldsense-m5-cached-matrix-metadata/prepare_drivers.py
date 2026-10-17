from pathlib import Path
import ast,json,shutil
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-request-residency';seed=w.parent/'worldsense-lazy-defaults'
for cohort in ['speed','confirmation']:(w/cohort).mkdir()
for name in ['freeze.py','prepare_cohort.py']:
 text=(old/name).read_text().replace('worldsense-request-residency','worldsense-cached-matrix-metadata')
 if name=='freeze.py':
  a=text.index('assert {k:(flags');b=text.index("(w/'build-flag-comparison.json')",a)
  text=text[:a]+"assert {k:(flags['off'][k],flags['on'][k])for k in flags['off']if flags['off'][k]!=flags['on'][k]}=={'GARNET_METAL_LAZY_RESIDENT_DEFAULTS':('OFF','ON'),'GARNET_METAL_CACHED_MATRIX_METADATA':('OFF','ON')}\n"+text[b:]
  text=text.replace("only declared deferred-storage/workspace flags differ",'only declared lazy-retirement/immutable-matrix-metadata flags differ')
  text=text.replace("s=(w.parent/'worldsense-vlm-workspace/controller.py').read_text().replace('worldsense-vlm-workspace','worldsense-cached-matrix-metadata')", "s=(w.parent/'worldsense-request-residency/controller.py').read_text().replace('worldsense-request-residency','worldsense-cached-matrix-metadata')")
  # The newer controller already includes LRPC readiness and power metadata.
  a=text.index(';s=s.replace(',text.index("s=(w.parent/"));b=text.index(";(w/'controller.py').write_text(s)",a);text=text[:a]+text[b:]
 else:
  a=text.index("assert json.loads((w/'private-trained");b=text.index("subprocess.run(['python3',str(w/'freeze.py')",a)
  text=text[:a]+"assert json.loads((w/'api-correctness-result.json').read_text())['four_input_six_input_vision_and_faults_pass']\nassert json.loads((w/'fault-correctness-result.json').read_text())['partial_allocation_failure_exact_recovery']\n"+text[b:]
 (w/name).write_text(text)
for cohort in ['speed','confirmation']:
 for name in ['prepare.py','run_stage.py']:(w/cohort/name).write_text((old/cohort/name).read_text().replace('worldsense-request-residency','worldsense-cached-matrix-metadata'))
# Keep all existing quantitative comparisons, golden/token/semantic assertions and
# source/library/request identities; replace only experiment-specific native route.
helper='''def native_route(call,enabled):
 t=call['timings']
 return ('cached_matrix_metadata' in call['backend'])==enabled and ('lazy_resident_defaults' in call['backend'])==enabled and (t['lazy_default_retired_bytes']>0)==enabled and (t['lazy_default_planned_slots']>0)==enabled and t['lazy_default_rehydrated_bytes']==0 and t['memory_after_decode']['task_info_status']==0

'''
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
 text=(old/rel).read_text().replace('worldsense-request-residency','worldsense-cached-matrix-metadata');tree=ast.parse(text)
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='residency_valid');lines=text.splitlines(True);text=''.join(lines[:fn.lineno-1])+helper+''.join(lines[fn.end_lineno:])
 if rel=='analyze_resource.py':
  a=text.index("t=c['timings'];pooled=");b=text.index(";memory=t['memory_after_decode']",a);text=text[:a]+"t=c['timings'];native=native_route(c,mode=='on')"+text[b:];text=text.replace("avoided_descriptor_bytes=t['resident_avoided_storage_bytes']","retired_default_bytes=t['lazy_default_retired_bytes']")
 else:
  # Each native route block starts on timings= and ends immediately before the serving source assertion.
  lines=text.splitlines(True);new=[];skip=False
  for line in lines:
   if "timings=c['timings'];pooled="in line:
    indent=line[:len(line)-len(line.lstrip())];new.append(indent+"timings=c['timings'];native=native_route(c,bool(expected))\n");skip=True;continue
   if skip:
    if "assert Path(c['timings']['serving_source'])"in line:skip=False;new.append(line)
    elif not line.lstrip().startswith('native='):
     # confirmation native expression can share subsequent semantic checks on the same line
     raise RuntimeError('Unexpected native block: '+line)
    continue
   new.append(line)
  text=''.join(new)
 (w/rel).write_text(text)
# Original resource wrapper is deliberately small: always normal restoration.
(w/'run_resource.py').write_text('''from pathlib import Path
import subprocess,sys
w=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(w/'prepare_cohort.py')],check=True)
try:
 subprocess.run([sys.executable,str(w/'controller.py'),'control-1','candidate-1'],check=True)
 subprocess.run([sys.executable,str(w/'analyze_resource.py')],check=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
''')
for p in w.rglob('*.py'):ast.parse(p.read_text())
print('Prospective resource/speed/full-confirmation drivers prepared; not dispatched')
