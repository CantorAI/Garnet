from pathlib import Path
import shutil,json,hashlib,subprocess,difflib
root=Path('/Users/shawn/CantorAI');w=Path(__file__).resolve().parent;d=root/'Garnet/results/20261016/worldsense-m5-private-execution';source=root/'out/research/worldsense-private-execution/source';fault=source.parent/'fault-source';repo=root/'Garnet';base=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();manifest=json.loads((w/'starting-source-manifest.json').read_text());patch=[];different=[]
for rel,h in manifest.items():
 p=source/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 cp=subprocess.run(['git','-C',str(repo),'show',base+':'+rel],capture_output=True)
 if cp.returncode==0 and cp.stdout==p.read_bytes():continue
 different.append(rel);target=d/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 try:old=cp.stdout.decode()if cp.returncode==0 else '';new=p.read_text()
 except UnicodeError:raise RuntimeError('Nontext overlay requires explicit binary support: '+rel)
 patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel if cp.returncode==0 else '/dev/null',tofile='b/'+rel))
(d/'source-overlay.patch').write_text(''.join(patch));subprocess.run(['git','-C',str(repo),'apply','--check',str(d/'source-overlay.patch')],check=True)
for rel in ['cmake/MetalWorldSense4BPrefix.cmake','test2026/metal/inventory_test.cpp','test2026/metal/memory_stats_test.mm','test2026/metal/prefix_copy_test.cpp','test2026/metal/matmul_test.cpp']:
 p=source/rel
 if p.exists():t=d/'source'/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
for name in ['fixture-1-8','fixture-2-64']:shutil.copytree(w/name,d/name,dirs_exist_ok=True)
fpatch=[];fchanges=[]
for p in sorted(fault.rglob('*')):
 if not p.is_file()or '__pycache__'in p.parts:continue
 rel=str(p.relative_to(fault));q=source/rel
 if q.exists()and q.read_bytes()==p.read_bytes():continue
 fchanges.append(rel);t=d/'fault-source'/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
 fpatch.extend(difflib.unified_diff(q.read_text().splitlines(True)if q.exists()else [],p.read_text().splitlines(True),fromfile='a/'+rel if q.exists()else '/dev/null',tofile='b/'+rel))
(d/'fault-overlay.patch').write_text(''.join(fpatch));assert sorted(fchanges)==['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h']
assert d==root/'Garnet/results/20261016/worldsense-m5-private-execution'
shutil.copy2(w/'f32-fallback.safetensors',d/'f32-fallback.safetensors')
for cohort in ['speed','confirmation']:
 if (w/cohort).exists():
  for p in (w/cohort).rglob('*'):
   if p.is_file() and p.suffix in ['.json','.log','.py','.csv','.txt']:
    t=d/cohort/p.relative_to(w/cohort);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
for p in d.rglob('__pycache__'):shutil.rmtree(p)
(d/'source-reconstruction.json').write_text(json.dumps(dict(base_commit=base,base_choice='Current committed tree selected after experiment; compatible reconstruction base, not source checkout identity at build time',changed_files=different,full_closure_sha256_file='starting-source-manifest.json',overlay_apply_check=True,fault_overlay_changed_files=fchanges,scope='Overlay covers every frozen source file different from base. Archives omit unchanged base files. Test fixtures contain original absolute paths; regenerate numeric fixtures with make_resident_fixtures.py when relocating.'),indent=2))
subprocess.run(['python3',str(repo/'docs/research/export_worldsense_calls.py'),str(d),str(d/'calls.csv')],check=True)
subprocess.run(['python3',str(d/'export_execution_routes.py'),str(d),str(d/'execution-routes.csv')],check=True)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2));print('Overlay apply check, source closure, separate fault delta, fixtures and calls exported',len(different))
