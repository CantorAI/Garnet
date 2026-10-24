from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');folder=r/'results/nvfp4-down-full-qualification'
wf=json.loads((folder/'workflow.json').read_text(encoding='utf8'));assert wf['status']=='failed' and len(wf['stages'])==2
for name,code in [('baseline',0),('state',1)]:
 rows=[json.loads(l) for l in (folder/(name+'-ledger/commands.jsonl')).read_text(encoding='utf8').splitlines()];assert rows[-1]['event']=='finished' and rows[-1]['exit_code']==code
state=json.loads((folder/'state/joined-report.json').read_text(encoding='utf8'));assert state['status']=='failed' and len(state['cases'])==6
assert 'Same-profile full packed output differs' in state['error'] and not (folder/'candidate').exists()
assert (folder/'state-difference-diagnostic.json').is_file()
files=[p for p in folder.rglob('*') if p.is_file()]
files += [r/n for n in ['run-nvfp4-down-full-qualification.py','diagnose-nvfp4-down-state.py','archive-nvfp4-down-state-failure.py','test-nvfp4-down-full-baseline.bat','test-nvfp4-down-full-state.bat','test-nvfp4-down-full-candidate.bat','study-state.json','preparation-manifest.json','results/nvfp4-down-runtime-attempt2/identity.json']]
files += [r/'Garnet/test2026/cuda/qwen38_5090'/n for n in ['probe_joined_decoder_full.py','probe_joined_program_equivalence.py']]
links={}
for name in ('joined-full-evidence.zip','nvfp4-down-runtime-evidence.zip'):links[name]=hashlib.sha256((r/'results'/name).read_bytes()).hexdigest()
assert links['joined-full-evidence.zip']=='b3dbb9267baa35a197b28a47ed5a27202782e5dc598650a5dd71fdd2c6adaa98'
assert links['nvfp4-down-runtime-evidence.zip']=='9778245cdc4f336627513063402b73a79bc79ecc3999adde7fb29d7e400b532e'
target=r/'results/nvfp4-down-state-failure-evidence.zip';assert not target.exists();digests={}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  name=p.relative_to(r).as_posix();raw=p.read_bytes();z.writestr(name,raw);digests[name]=hashlib.sha256(raw).hexdigest()
 z.writestr('evidence-manifest.json',json.dumps({'files':digests,'linked_archives':links,'scope':'Preserved option-on full-state exactness failure after rebuilt option-off12 target/vector goldens passed. First packed chunks retained; remaining full-state hashes are producer assertions. No candidate generation/MTP/performance admission.'},indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(digests)}))
