"""Archive initial validation by source overlay only; execute after all timing."""
from pathlib import Path
import json,hashlib,shutil,difflib,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');d=r/'Garnet/results/20261016/worldsense-m5-request-residency';assert d.name=='worldsense-m5-'+w.name.removeprefix('worldsense-')
s=r/'out/research/worldsense-request-residency/source';old=s.parent/'pre-recovery-source';identity=json.loads((w/'pre-recovery-fix/initial-validation-identity.json').read_text());patch=[];changed=[]
for rel,h in identity['all_source_sha256'].items():
 p=old/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 if p.read_bytes()==(s/rel).read_bytes():continue
 changed.append(rel);target=d/'pre-recovery-source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 patch.extend(difflib.unified_diff((s/rel).read_text().splitlines(True),p.read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
assert changed==['src/backends/metal/decoder_session.cpp'],changed
(d/'pre-recovery-overlay.patch').write_text(''.join(patch));subprocess.run(['git','-C',str(s),'apply','--check',str(d/'pre-recovery-overlay.patch')],check=True)
for p in (w/'pre-recovery-fix').rglob('*'):
 if p.is_file()and p.suffix in ['.json','.log','.txt','.make']:
  target=d/'pre-recovery-fix'/p.relative_to(w/'pre-recovery-fix');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
(d/'pre-recovery-reconstruction.json').write_text(json.dumps({'base':'Final frozen WS33 source reconstructed from source-overlay.patch','overlay':'pre-recovery-overlay.patch','full_initial_closure_sha256':'pre-recovery-fix/initial-validation-identity.json','changed_files':changed,'apply_check_pass':True,'scope':'Initial native/fullstate validation passed before static discovery of poisoned-Begin recovery gap. Alltiming uses correctedsource;no favorable unchanged retry.'},indent=2));print('Initial source/results identity archived by one-file overlay')
