from pathlib import Path
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/Garnet');d=r/'results/20261016/worldsense-m5-private-resident-dense512';b=r.parent/'out/build';snapshot={}
subprocess.run(['python3',str(r/'docs/research/export_worldsense_calls.py'),str(d/'speed'),str(d/'speed/calls.csv')],check=True)
for mode in ['off','on','fault']:
 p=b/('garnet-metal-worldsense-private-resident-dense512-'+mode);files=[p/'libgarnet.dylib',p/'CMakeCache.txt']+sorted(p.glob('garnet_*_test'));snapshot[mode]={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files if f.is_file()}
 if mode in ['off','on']:assert (p/'dense_decode1024_words.bin').read_bytes()==(w/('dense-words-'+mode+'.bin')).read_bytes()
(d/'post-test-build-identities.json').write_text(json.dumps({'scope':'Post-test identity snapshot; predispatch freeze recordedseparately.','files':snapshot},indent=2))
for name in ['verify_archive.py','post_finalize.py']:shutil.copy2(w/name,d/name)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2))
subprocess.run(['python3',str(w/'verify_archive.py')],check=True)
subprocess.run(['python3',str(r/'docs/research/build_metal_method_index.py')],check=True)
subprocess.run(['git','-C',str(r),'diff','--check'],check=True)
assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain','--untracked-files=all','--','results/20261016/worldsense-m5-dense-decode512'],text=True)
