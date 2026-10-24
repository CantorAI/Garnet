from pathlib import Path
import hashlib,json,subprocess
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/nsight-2026-tool-setup-short';out.mkdir(exist_ok=False)
prior=json.loads((r/'results/nsight-2026-tool-setup/setup.json').read_text(encoding='utf8'));package=r/'tools/NsightSystems-2026.5.1.161.msi';destination=r/'tools/n26'
assert prior['extraction_exit_code']==1603 and prior['authenticode']['Status']=='Valid' and 'NVIDIA Corporation' in prior['authenticode']['Subject']
assert hashlib.sha256(package.read_bytes()).hexdigest()==prior['package_sha256']=='379c0a15a9cf7b8028081073fdd1d9798b9aaa618fb466c6a01785b69ed01d5f2'
assert not destination.exists(),'Inspect fixed extraction target; no overwrite or duplicate package download'
report={'status':'extracting_short_path','package_sha256':prior['package_sha256'],'authenticode':prior['authenticode'],'target':str(destination),'scope':'Same signed MSI administrative image at shorter fixed tools/n26 path after error1304 at260-character file path; no global install, runtime/driver change or restart.'}
def save():(out/'setup.json').write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
 code=subprocess.run(['msiexec.exe','/a',str(package),'/qn','/norestart','TARGETDIR='+str(destination),'/L*v',str(out/'administrative-extraction.log')]).returncode
 report['extraction_exit_code']=code;save();assert code==0,'Preserve failure; no global installation fallback'
 candidates=[p for p in destination.rglob('nsys.exe') if 'target-windows-x64' in p.parts];assert len(candidates)==1
 tool=candidates[0];version=subprocess.check_output([str(tool),'--version'],text=True);assert '2026.5.1' in version
 report.update(status='extracted_cli_version_verified',nsys_path=str(tool),nsys_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),version=version);save();print(json.dumps(report,indent=2))
except Exception as error:report.update(status='failed',error=str(error));save();raise
