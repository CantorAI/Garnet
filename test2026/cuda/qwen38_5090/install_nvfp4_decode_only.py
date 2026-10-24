from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023')
processes=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'xlang3.exe'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"]).decode() or '[]')
if isinstance(processes,dict):processes=[processes]
assert not any(str(r).lower() in (p.get('CommandLine') or '').lower() for p in processes),'Study runtime still live'
state=json.loads((r/'study-state.json').read_text());prior=json.loads((r/'preparation-manifest.json').read_text())
records={x['path']:x for x in json.loads((r/'source-manifest.json').read_text())['files']};records.update({x['path']:x for x in prior})
for name,row in records.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==row['sha256'],name
dll=r/'build/bin/garnet.dll';digest=hashlib.sha256(dll.read_bytes()).hexdigest()
assert digest=='e82279af698dd7798be3f158ace999eef29be4ed609e8cd50cf1ae36c90cd750'
backup=r/'build/qualified-before-nvfp4-decode-only';backup.mkdir(exist_ok=True)
if (backup/'garnet.dll').exists():assert hashlib.sha256((backup/'garnet.dll').read_bytes()).hexdigest()==digest
else:shutil.copy2(dll,backup/'garnet.dll')
for name in ('study-state.json','preparation-manifest.json'):
 target=backup/name
 if not target.exists():shutil.copy2(r/name,target)
with zipfile.ZipFile(r/'nvfp4-decode-only-source.zip') as z:
 assert z.testzip() is None;patch=json.loads(z.read('nvfp4-decode-only-patch.json'));assert state['commit']==patch['base']
 for row in patch['files']:
  assert row['path'].startswith(('Garnet/src/backends/tensorrt/','Garnet/test2026/cuda/qwen38_5090/','Garnet/docs/research/')) and not row['path'].endswith('CMakeLists.txt')
  name=row['path'];assert (r/name).resolve().is_relative_to(r.resolve())
  raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==row['sha256']
  target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
  if (r/name).exists() and not target.exists():shutil.copy2(r/name,target)
 for row in patch['files']:(r/row['path']).write_bytes(z.read(row['path']))
 updates={row['path']:row for row in prior};updates.update({row['path']:row for row in patch['files']})
 state.update(commit=patch['commit'],tracked_updates=sorted(set(state['tracked_updates'])|{row['path'].removeprefix('Garnet/') for row in patch['files']}))
 (r/'preparation-manifest.json').write_text(json.dumps(list(updates.values()),indent=2));(r/'study-state.json').write_text(json.dumps(state,indent=2))
 result={'old_source_files_verified':len(records),'source_commit':patch['commit'],'native_build_required':True,'standalone_probe_build_required':True,'compiled_native_commit':'5d675933663536f27f36bcb8e9566ced3dbc7656','previous_dll_sha256':digest,'patch':patch}
 (r/'results/nvfp4-decode-only-source-install.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
