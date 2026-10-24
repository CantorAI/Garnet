from pathlib import Path
import hashlib,json,subprocess
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/layer0-stage-probe-fix-install';assert not out.exists()
events=[json.loads(l) for l in (r/'results/layer0-real-stage-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==1
old_archive=r/'results/layer0-real-stage-attempt1-evidence.zip';archive_hash=hashlib.sha256(old_archive.read_bytes()).hexdigest();assert archive_hash=='65cb85f337c943a1025fe67f6201bb5a150087fea15a68b06d4723d153cccfc1'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
build=json.loads((r/'results/layer0-stage-build/manifest.json').read_text());assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()==build['library_sha256']
name='Garnet/test2026/cuda/qwen38_5090/probe_layer0_real_stages.py';dest=r/name;old=dest.read_bytes();assert hashlib.sha256(old).hexdigest()==build['source_files'][name]['sha256'];new=(r/'probe-layer0-real-stages-warmed.py').read_bytes();assert new!=old
out.mkdir();(out/'original-probe.py').write_bytes(old);dest.write_bytes(new)
(out/'receipt.json').write_text(json.dumps({'status':'test_setup_fixed_same_native_binary','library_sha256':build['library_sha256'],'failure_archive_sha256':archive_hash,'old_probe_sha256':hashlib.sha256(old).hexdigest(),'probe_sha256':hashlib.sha256(new).hexdigest(),'scope':'Only test input construction, explicit completed warmup and result release changed. Full primary/negative/repeat/projection/graph gates unchanged; no original data overwritten, no native rebuild or speed claim.'},indent=2));print('PROBE_SETUP_FIXED; native binary unchanged and original failure preserved')
