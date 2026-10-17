from pathlib import Path
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/Garnet');d=r/'results/20261016/worldsense-m5-physical-memory';b=r.parent/'out/build';snapshot={}
assert d.name=='worldsense-m5-'+w.name.removeprefix('worldsense-'), 'Archive destination must match experiment before any mutation'
if (d/'speed/control-1.json').exists():subprocess.run(['python3',str(r/'docs/research/export_worldsense_calls.py'),str(d/'speed'),str(d/'speed/calls.csv')],check=True)
for mode in ['off','on']:
 p=b/('garnet-metal-worldsense-physical-memory-'+mode);files=[p/'libgarnet.dylib',p/'CMakeCache.txt']+sorted(p.glob('garnet_*_test'));snapshot[mode]={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files if f.is_file()}
(d/'post-test-build-identities.json').write_text(json.dumps({'scope':'Post-test identity snapshot; predispatch freeze recordedseparately.','files':snapshot},indent=2))
for name in ['verify_archive.py','post_finalize.py']:shutil.copy2(w/name,d/name)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2))
subprocess.run(['python3',str(w/'verify_archive.py')],check=True)
subprocess.run(['python3',str(r/'docs/research/build_metal_method_index.py')],check=True)
subprocess.run(['git','-C',str(r),'diff','--check'],check=True)
assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain','--untracked-files=all','--','results/20261016/worldsense-m5-dense-decode512'],text=True)

assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain','--untracked-files=all','--','results/20261016/worldsense-m5-cached-matrix-metadata'],text=True)
