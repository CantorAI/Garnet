"""Extract authenticated Windows payload using read-only MSI mapping and existing7Zip."""
from pathlib import Path
import collections,hashlib,json,os,re,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
out=r/'results/nsight-2026-cab-extraction';out.mkdir(exist_ok=False);package=r/'tools/NsightSystems-2026.5.1.161.msi';receipt=json.loads((r/'results/nsight-2026-tool-setup/setup.json').read_text(encoding='utf8'))
assert receipt['authenticode']['Status']=='Valid' and 'NVIDIA Corporation' in receipt['authenticode']['Subject']
assert hashlib.sha256(package.read_bytes()).hexdigest()==receipt['package_sha256']
plan_path=r/'results/nsight-2026-cab-plan/plan.json';plan=json.loads(plan_path.read_text(encoding='utf8'));files=plan['files'];assert len(files)==4039
root=r/'tools/n26';cache=r/'tools/n26-cabs';stage=r/'tools/n26-stage';assert not cache.exists() and not stage.exists();assert not root.exists() or not any(v.is_file() for v in root.rglob('*'))
root.mkdir(exist_ok=True);cache.mkdir();stage.mkdir();seven=Path('C:/Program Files/7-Zip/7z.exe');assert seven.is_file()
groups=collections.defaultdict(list)
for row in files:
 assert re.fullmatch(r'[A-Za-z0-9_.]+',row['key']) and re.fullmatch(r'cab[0-9]+\.cab',row['cabinet'])
 destination=root/row['relative_path'];assert destination.resolve().is_relative_to(root.resolve()) and all(not any(ch in component for ch in '<>:"|?*') for component in Path(row['relative_path']).parts)
 groups[row['cabinet']].append(row)
report={'status':'extracting_cabinet_streams','package_sha256':receipt['package_sha256'],'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'scope':'Copy Windows payload from original signed MSI, preserving target/host relative layout inside fixed tools/n26. No MSI/global install, driver/runtime change or restart.'}
def save():(out/'setup.json').write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
 code=run([str(seven),'e',str(package),*sorted(groups),'-o'+str(cache),'-y'],out/'cabinet-stream-ledger','setup','extract-authenticated-msi-cabinet-streams',[seven,package,plan_path]);assert code==0
 hashes={}
 for cabinet,rows in sorted(groups.items()):
  selection=out/(cabinet+'-selection.txt');selection.write_text('\n'.join(row['key'] for row in rows)+'\n',encoding='utf8')
  code=run([str(seven),'e',str(cache/cabinet),'-o'+str(stage),'-i@'+str(selection),'-scsUTF-8','-y'],out/'payload-ledger','setup','extract-windows-'+cabinet,[seven,cache/cabinet,selection]);assert code==0
  expected={v['key'] for v in rows};assert {v.name for v in stage.iterdir()}==expected,'Unexpected payload members; preserve before any move'
  for row in rows:
   source=stage/row['key'];destination=root/row['relative_path'];assert source.is_file() and source.stat().st_size==row['bytes'] and not destination.exists()
   assert source.resolve().is_relative_to(stage.resolve()) and destination.resolve().is_relative_to(root.resolve())
   destination.parent.mkdir(parents=True,exist_ok=True);os.replace(source,destination);hashes[row['relative_path']]=hashlib.sha256(destination.read_bytes()).hexdigest()
  print('WINDOWS_PAYLOAD_FILES_MOVED',len(hashes),flush=True)
 (out/'payload-sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
 tool=root/'target-windows-x64/nsys.exe';assert tool.is_file()
 cmd="$s=Get-AuthenticodeSignature -LiteralPath 'D:/CantorAI/5090-study-20261023/tools/n26/target-windows-x64/nsys.exe';[pscustomobject]@{Status=$s.Status.ToString();Subject=$s.SignerCertificate.Subject;Thumbprint=$s.SignerCertificate.Thumbprint}|ConvertTo-Json -Compress"
 signature=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True));report['nsys_authenticode']=signature;save();assert signature['Status']=='Valid' and 'NVIDIA Corporation' in signature['Subject']
 version=subprocess.check_output([str(tool),'--version'],text=True);assert '2026.5.1' in version
 report.update(status='windows_payload_and_cli_version_verified',files_verified=len(hashes),nsys_path=str(tool),nsys_sha256=hashes['target-windows-x64/nsys.exe'],version=version);save();print(json.dumps(report,indent=2))
except Exception as error:report.update(status='failed',error=str(error));save();raise
