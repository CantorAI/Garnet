from pathlib import Path
import json,hashlib,zipfile
r=Path('D:/CantorAI/5090-study-20261023');o=r/'results/operand-sweep-attempt2'
for p in o.glob('*-ledger/commands.jsonl'):
 rows=[json.loads(x) for x in p.read_text().splitlines()];assert rows[-1]['event']=='finished',str(p)
files=[p for p in o.rglob('*') if p.is_file()]
files += [r/x for x in ['run-operand-sweep.py','install-int4-reference.py','build-operand-swap.bat','int4-reference-source.zip','study-state.json','preparation-manifest.json','results/int4-reference-source-install.json','Garnet/test2026/cuda/qwen38_5090/nvfp4_operand_swap_test.cu','Garnet/src/backends/tensorrt/modelopt_cublaslt.cpp','Garnet/src/cuda/modelopt_nvfp4.cu']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};target=r/'results/operand-sweep-attempt2-evidence.zip';assert not target.exists()
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in raw.items():z.writestr(name,data)
 z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'scope':'Standalone synthetic operand-swap experiment, including failures. No model performance claim.'},indent=2))
print(json.dumps({'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'entries':len(raw)+1}))
