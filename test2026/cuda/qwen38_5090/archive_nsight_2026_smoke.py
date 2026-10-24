from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');d=r/'results/nsight-cuda-smoke-2026-attempt1';w=json.loads((d/'workflow.json').read_text());assert w['profile_exit_code']==w['stats_exit_code']==0
names=['nsight-2026-tool-setup','nsight-2026-tool-setup-ledger','nsight-2026-tool-setup-short','nsight-2026-tool-setup-short-ledger','nsight-2026-tool-setup-short-fix','nsight-2026-tool-setup-short-fix-ledger','nsight-2026-tool-setup-direct','nsight-2026-tool-setup-direct-ledger','nsight-2026-cab-plan','nsight-2026-cab-extraction','nsight-2026-cab-extraction-ledger','nsight-cuda-smoke-2026-attempt1']
files=[p for name in names for p in (r/'results'/name).rglob('*') if p.is_file()]
files += [r/n for n in ['setup-nsight-2026.py','extract-nsight-2026-short.py','extract-nsight-2026-short-fix.py','extract-nsight-2026-direct.py','read-nsight-msi-directories.py','plan-nsight-cab-extraction.py','extract-nsight-2026-cabs.py','run-nsight-cuda-smoke-2026.py','test-nsight-cuda-smoke-2026.bat','archive-nsight-2026-smoke.py','Garnet/test2026/cuda/qwen38_5090/probe_trt_layer_profiler.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/nsight-cuda-smoke-2026-attempt1-evidence.zip';assert not out.exists()
manifest={'scope':'Side-by-side signed-package extraction receipts, preserved setup failures and actual bounded CUDA smoke trace. Tool payload binaries remain in fixed remote tools/n26; payload hash manifest/signature fields are producer records, not independently retained tool bytes. No model performance admission.','linked_archives':{'native-4b-http-evidence.zip':hashlib.sha256((r/'results/native-4b-http-evidence.zip').read_bytes()).hexdigest()},'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw)}))
