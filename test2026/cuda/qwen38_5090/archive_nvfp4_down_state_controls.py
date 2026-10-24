from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');folder=r/'results/nvfp4-down-state-controls';report=json.loads((folder/'control-comparison.json').read_text(encoding='utf8'))
assert report['status']=='controls_completed' and report['all_inputs_equal'] and report['default_original_equal_old_teacher'] and report['default_joined_equal_original'] and not report['active_candidate_equal_default_joined']
for name in ('original','joined'):
 events=[json.loads(l) for l in (r/('results/nvfp4-down-control-'+name+'-ledger/commands.jsonl')).read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
files=[p for p in folder.rglob('*') if p.is_file()]
for name in ('original','joined'):
 for suffix in ('ledger','resources'):files += [p for p in (r/('results/nvfp4-down-control-'+name+'-'+suffix)).rglob('*') if p.is_file()]
files += [r/n for n in ['run-nvfp4-down-state-controls.py','test-nvfp4-down-control-original.bat','test-nvfp4-down-control-joined.bat','archive-nvfp4-down-state-controls.py','study-state.json','preparation-manifest.json','Garnet/test2026/cuda/qwen38_5090/probe_joined_program_equivalence.py']]
target=r/'results/nvfp4-down-state-controls-evidence.zip';assert not target.exists();digests={};aliases={};stored={}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  name=p.relative_to(r).as_posix();raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();digests[name]=h
  if h in stored:aliases[name]=stored[h]
  else:z.writestr(name,raw);stored[h]=name
 z.writestr('evidence-manifest.json',json.dumps({'files':digests,'aliases':aliases,'failure_archive_sha256':hashlib.sha256((r/'results/nvfp4-down-state-failure-evidence.zip').read_bytes()).hexdigest(),'scope':'Byte-identical logical snapshots stored once with explicit SHA-verified aliases; original raw files retained on study PC. Both option-off controls match old teacher and each other; failed option-on differs. No internal kernel-cause, harmlessness, candidate performance or2x claim.'},indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'logical_files':len(digests),'stored_files':len(stored),'aliases':len(aliases)}))
