from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/layer0-stage-source-install';assert not out.exists()
events=[json.loads(l) for l in (r/'results/native-controls-abba-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
proof=json.loads((r/'results/native-controls-abba-verified.json').read_text());assert proof['status']=='four_leg_controls_abba_evidence_verified' and proof['archive_sha256']==hashlib.sha256((r/'results/native-controls-abba-evidence.zip').read_bytes()).hexdigest()
resources=json.loads((r/'results/deployed-io-resources-attempt1/corrected-parse.json').read_text());assert resources['status']=='deployed_sm120_static_resources_read'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(r).free>2*(1<<30),'Inspect free disk before native build'
# The study itself must have no remaining Xlang worker; unrelated installs stay untouched.
query="@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'xlang3.exe' -and $_.CommandLine -like '*5090-study-20261023*'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
assert not json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',query]) or b'[]')
library=r/'build/bin/garnet.dll';old=hashlib.sha256(library.read_bytes()).hexdigest();assert old==resources['library_sha256']=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b';backup=r/'build/bin/garnet-before-layer0-stages.dll'
if backup.exists():assert hashlib.sha256(backup.read_bytes()).hexdigest()==old
with zipfile.ZipFile(r/'layer0-stage-source.zip') as z:
 m=json.loads(z.read('manifest.json'))
 for n,v in m.items():
  dest=(r/n).resolve();assert dest.is_relative_to((r/'Garnet').resolve()) and hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
  if 'previous_sha256' in v:assert hashlib.sha256(dest.read_bytes()).hexdigest()==v['previous_sha256'],n
  else:assert not dest.exists(),n
 out.mkdir()
 for n,v in m.items():
  dest=r/n
  if 'previous_sha256' in v:
   saved=out/'backup'/n;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(dest.read_bytes())
 if not backup.exists():shutil.copyfile(library,backup)
 assert hashlib.sha256(backup.read_bytes()).hexdigest()==old
 for n,v in m.items():(r/n).write_bytes(z.read(n));assert hashlib.sha256((r/n).read_bytes()).hexdigest()==v['sha256']
 (out/'receipt.json').write_text(json.dumps({'status':'source_installed_binary_not_rebuilt','old_library_sha256':old,'backup':str(backup),'package_sha256':hashlib.sha256((r/'layer0-stage-source.zip').read_bytes()).hexdigest(),'files':m},indent=2))
print('BOUNDED_LAYER0_SOURCE_INSTALLED; previous source andDLL preserved; no GPU worker started')
