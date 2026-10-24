from pathlib import Path
import hashlib,json,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');d=r/'results/warm-mtp-resident-controls-attempt1';outer=r/'results/warm-mtp-resident-controls-controller-ledger'
events=[json.loads(l) for l in (outer/'commands.jsonl').read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished','Observe original controller handle terminal before archiving'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024,'Inspect live GPU process; do not snapshot or start competing model'
files=[p for directory in (d,outer) for p in directory.rglob('*') if p.is_file()]
files += [r/n for n in ['run-warm-mtp-resident-controls.py','probe-warm-mtp-resident-controls.py','test-warm-mtp-resident-controls.bat','test-resident-controls-transport.bat','archive-warm-mtp-resident-controls.py']]
files += [r/'Garnet'/n for n in ['test2026/cuda/qwen38_5090/run_logged.py','test2026/cuda/qwen38_5090/probe_shared_trt_layers.py','xModel/qwen35/vl_0_8b/cuda_speculative.py','xModel/qwen35/vl_0_8b/cuda_packed_replay.py','xModel/qwen35/vl_0_8b/cuda_packed_decoder.py','xModel/qwen35/vl_0_8b/cuda_packed_program.py','xModel/qwen35/vl_0_8b/cuda_mtp.py','src/backends/tensorrt/cuda_tensor_io.cpp','src/backends/tensorrt/trt_builder.cpp','src/tensor/tensor_helper.cpp','src/cuda/qwen35_io.cu','test2026/cuda/test_qwen35_resident_controls.py','xModel/qwen35/vl_0_8b/cuda_hybrid.py','xModel/qwen35/vl_0_8b/cuda_verify.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};anchors={n:hashlib.sha256((r/'results'/n).read_bytes()).hexdigest() for n in ['native-mtp3-decode-only-off-http-evidence.zip','warm-mtp-trace-unprofiled-control-evidence.zip','nvfp4-decode-only-full-evidence.zip','warm-mtp-cuda-trace-2026-attempt1-evidence.zip','warm-mtp-cuda-trace-startup-attempt1-evidence.zip']}
manifest={'scope':'Optional resident request controls transport/full-model correctness, including failure or success. No timing, HTTP or2x claim. Independent validation required.','terminal':events[-1],'linked_archives':anchors,'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
out=r/'results/warm-mtp-resident-controls-attempt1-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw)}))
