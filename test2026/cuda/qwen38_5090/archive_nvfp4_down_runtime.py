from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/nvfp4-down-runtime-attempt2'
done=json.loads((out/'workflow-passed.json').read_text(encoding='utf8'));assert done['status']=='component_gates_and_native_build_passed'
for name in ('build-tests','contracts','legacy-plugin','oracle','checkpoint','build-native'):
 rows=[json.loads(l) for l in (out/(name+'-ledger/commands.jsonl')).read_text(encoding='utf8').splitlines()];assert rows[-1]['event']=='finished' and rows[-1]['exit_code']==0
failed=r/'results/nvfp4-down-runtime-attempt1';rows=[json.loads(l) for l in (failed/'build-tests-ledger/commands.jsonl').read_text(encoding='utf8').splitlines()];assert rows[-1]['exit_code']==1
files=[p for folder in (out,failed) for p in folder.rglob('*') if p.is_file()]
files += [r/n for n in ['run-nvfp4-down-runtime.py','run-nvfp4-down-runtime-fix.py','install-nvfp4-down-runtime.py','install-nvfp4-down-runtime-fix.py','archive-nvfp4-down-runtime.py','build-nvfp4-down-runtime-tests.bat','build-nvfp4-down-runtime-native.bat','nvfp4-down-runtime-source.zip','nvfp4-down-runtime-fix-source.zip','study-state.json','preparation-manifest.json','results/nvfp4-down-runtime-source-install.json','results/nvfp4-down-runtime-fix-source-install.json']]
for name in ['modelopt_cublaslt.cpp','modelopt_cublaslt.h','modelopt_nvfp4_plugin.cpp','modelopt_nvfp4_plugin.h','modelopt_nvfp4_lowering.cpp']:files.append(r/'Garnet/src/backends/tensorrt'/name)
for name in ['modelopt_plugin_test.cpp','nvfp4_operand_swap_test.cu','monitor_native_probe.py']:files.append(r/'Garnet/test2026/cuda/qwen38_5090'/name)
prior=r/'results/nvfp4-down-candidate-attempt1-evidence.zip';prior_sha=hashlib.sha256(prior.read_bytes()).hexdigest();assert prior_sha=='44a374aa2b96d3316715f70aad5e44b1eeec1dbc6e71409439cf4f80c2cdb2ca'
target=r/'results/nvfp4-down-runtime-evidence.zip';assert not target.exists();digests={}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  raw=p.read_bytes();name=p.relative_to(r).as_posix();z.writestr(name,raw);digests[name]=hashlib.sha256(raw).hexdigest()
 z.writestr('evidence-manifest.json',json.dumps({'files':digests,'fixture_archive':'nvfp4-down-candidate-attempt1-evidence.zip','fixture_archive_sha256':prior_sha,'scope':'Default-off opt-in native selector/serialization/workspace component gates and rebuilt serving library. Original test compile failure retained. Full-model/MTP/HTTP/performance qualification separate and pending.'},indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(digests)}))
