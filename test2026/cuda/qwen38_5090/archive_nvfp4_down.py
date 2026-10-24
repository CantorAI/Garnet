from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/nvfp4-down-candidate-attempt1'
ledgers=list(out.glob('*-ledger/commands.jsonl'));assert ledgers
for p in ledgers:
 rows=[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()];assert rows[-1]['event']=='finished'
files=[p for p in out.rglob('*') if p.is_file()]
files += [r/n for n in ['run-nvfp4-down.py','install-nvfp4-down.py','archive-nvfp4-down.py','build-operand-swap.bat','nvfp4-down-source.zip','study-state.json','preparation-manifest.json','results/nvfp4-down-source-install.json','Garnet/test2026/cuda/qwen38_5090/nvfp4_operand_swap_test.cu','Garnet/test2026/cuda/qwen38_5090/prepare_nvfp4_down_fixture.py','Garnet/src/backends/tensorrt/modelopt_cublaslt.cpp','Garnet/src/backends/tensorrt/modelopt_nvfp4_lowering.cpp','models/Qwen3.8-27B-NVFP4/model.safetensors.index.json']]
assert all(p.is_file() for p in files)
target=r/'results/nvfp4-down-candidate-attempt1-evidence.zip';assert not target.exists()
digests={}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  name=p.relative_to(r).as_posix();raw=p.read_bytes();z.writestr(name,raw);digests[name]=hashlib.sha256(raw).hexdigest()
 z.writestr('evidence-manifest.json',json.dumps({'files':digests,'scope':'Actual full pinned layer0 down weights/scales; synthetic signed activations; complete raw baseline/candidate outputs if produced. Preserve terminal failures. No timing/model-speed claim.'},indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(digests)}))
