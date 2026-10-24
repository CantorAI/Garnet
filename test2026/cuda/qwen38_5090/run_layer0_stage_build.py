from pathlib import Path
import hashlib,json,shutil,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
out=r/'results/layer0-stage-build';assert not out.exists();receipt=r/'results/layer0-stage-source-install/receipt.json';installed=json.loads(receipt.read_text());assert installed['status']=='source_installed_binary_not_rebuilt'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
library=r/'build/bin/garnet.dll';assert hashlib.sha256(library.read_bytes()).hexdigest()==installed['old_library_sha256'];assert hashlib.sha256(Path(installed['backup']).read_bytes()).hexdigest()==installed['old_library_sha256']
for n,v in installed['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==v['sha256'],n
out.mkdir();artifacts=[r/'build-layer0-stages.bat',receipt]+[r/n for n in installed['files']]
code=run(['cmd.exe','/d','/c',str(r/'build-layer0-stages.bat')],out/'build-ledger','build','bounded-linear-stage-cpp-extension',artifacts)
if code:raise SystemExit(code)
new=hashlib.sha256(library.read_bytes()).hexdigest();assert new!=installed['old_library_sha256']
for n,v in installed['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==v['sha256'],n
copy=r/'build/bin/garnet-layer0-stage-attempt1.dll';assert not copy.exists();shutil.copyfile(library,copy);assert hashlib.sha256(copy.read_bytes()).hexdigest()==new
(out/'manifest.json').write_text(json.dumps({'status':'native_build_completed_unqualified','old_library_sha256':installed['old_library_sha256'],'library_sha256':new,'preserved_candidate':str(copy),'source_install_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'source_files':installed['files'],'scope':'Successful native build/link only. No stage/default numeric, fullmodel, serving or speed admission.'},indent=2));print('LAYER0_NATIVE_BUILD_CLOSED; correctness gates pending',flush=True)
