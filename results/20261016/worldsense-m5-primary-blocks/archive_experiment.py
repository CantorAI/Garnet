from pathlib import Path
import json,hashlib,shutil,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/Garnet');s=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-blocks/cpu-source');d=r/'results/20261016/worldsense-m5-primary-blocks'
assert d.name=='worldsense-m5-'+w.name.removeprefix('worldsense-') and not d.exists()
d.mkdir(parents=True)
for p in w.iterdir():
 if p.is_file()and p.suffix in ['.json','.txt','.cpp','.py','.log']:shutil.copy2(p,d/p.name)
shutil.copytree(s,d/'source/cpu-source')
selection=json.loads((w/'matrix-selection.json').read_text());p=Path(selection['inventory_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==selection['inventory_sha256'];shutil.copy2(p,d/'source-inventory.json')
body=(w/'archive-body.txt').read_text();(d/'README.md').write_text('# WS-M5-36: sole-primary lossless BF16 blocks — CPU feasibility\n\n'+body)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2))
subprocess.run(['python3',str(w/'verify_archive.py')],check=True)
subprocess.run(['python3',str(r/'docs/research/build_metal_method_index.py')],check=True)
subprocess.run(['git','-C',str(r),'diff','--check'],check=True)
print('CPU feasibility archive complete',d)
