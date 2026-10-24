"""Download NVIDIA-signed profiler and administratively extract into fixed study tools."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,urllib.request
from urllib.parse import urlsplit,urlunsplit
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/nsight-2026-tool-setup';out.mkdir(exist_ok=False)
url='https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_5/nsightsystems-2026.5.1.161-3889610.msi'
package=r/'tools/NsightSystems-2026.5.1.161.msi';partial=package.with_suffix('.msi.part');destination=r/'tools/nsight-systems-2026.5.1'
assert not package.exists() and not partial.exists() and not destination.exists(),'Inspect existing fixed tool assets before resuming; do not duplicate/overwrite'
assert shutil.disk_usage(r).free>5*(1<<30)
report={'status':'downloading','source_url':url,'scope':'Side-by-side administrative extraction of signed NVIDIA tool into study folder. No global install, driver/toolkit change, application-source change or restart.'}
def save():(out/'setup.json').write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
 digest=hashlib.sha256();total=0
 with urllib.request.urlopen(url,timeout=60) as response,partial.open('xb') as target:
  final=urlsplit(response.geturl());report['resolved_source_without_query']=urlunsplit((final.scheme,final.netloc,final.path,'',''))
  while True:
   block=response.read(4<<20)
   if not block:break
   if total==0:assert block[:8]==bytes.fromhex('d0cf11e0a1b11ae1'),'Response is not an MSI compound file'
   target.write(block);digest.update(block);total+=len(block)
   assert total<2*(1<<30),'Unexpected package size; retain partial and inspect'
   if total%(32<<20)==0:print('NVIDIA_PROFILER_DOWNLOAD_BYTES',total,flush=True)
 assert total>1<<20;partial.rename(package);report.update(status='checking_signature',package_bytes=total,package_sha256=digest.hexdigest());save()
 command="$s=Get-AuthenticodeSignature -LiteralPath 'D:/CantorAI/5090-study-20261023/tools/NsightSystems-2026.5.1.161.msi';[pscustomobject]@{Status=$s.Status.ToString();Subject=$s.SignerCertificate.Subject;Thumbprint=$s.SignerCertificate.Thumbprint}|ConvertTo-Json -Compress"
 signature=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True));report['authenticode']=signature;save();assert signature['Status']=='Valid' and 'NVIDIA Corporation' in signature['Subject'],'Do not execute an unverified package'
 report['status']='extracting_administrative_image';save()
 code=subprocess.run(['msiexec.exe','/a',str(package),'/qn','/norestart','TARGETDIR='+str(destination),'/L*v',str(out/'administrative-extraction.log')]).returncode
 report['extraction_exit_code']=code;save();assert code==0,'No automatic restart or global installation fallback'
 candidates=list(destination.rglob('nsys.exe'));cli=[p for p in candidates if 'target-windows-x64' in p.parts];assert len(cli)==1,'Inspect extracted CLI layout'
 tool=cli[0];version=subprocess.check_output([str(tool),'--version'],text=True);assert '2026.5.1' in version
 report.update(status='extracted_cli_version_verified',nsys_path=str(tool),nsys_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),version=version,free_bytes_after=shutil.disk_usage(r).free);save();print(json.dumps(report,indent=2))
except Exception as error:report.update(status='failed',error=str(error));save();raise
