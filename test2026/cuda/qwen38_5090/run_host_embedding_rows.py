"""Run isolated selected-row embedding/lifecycle gates on the new native build."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('D:/CantorAI/5090-study-20261023');methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda raw:hashlib.sha256(raw).hexdigest()
out=root/'results/host-embedding-rows-attempt1';assert not out.exists()
build_path=root/'results/native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_text())
assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
ledger=root/'results/native-host-embedding-build-attempt1/build-ledger/commands.jsonl'
events=[json.loads(line) for line in ledger.read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes())
assert digest==build['library_sha256']==sha(Path(build['retained_candidate_library']).read_bytes())
assert digest!=build['old_library_sha256']
for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
    for name,want in build[key].items():assert sha((root/'Garnet'/name).read_bytes())==want
proof_path=root/'results/batch-mtp-deferred-http-pilot-attempt1-verified.json'
assert sha(proof_path.read_bytes())==build['prior_http_proof_sha256']
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
out.mkdir();fixture=out/'fixture';native=out/'native'
workflow={'status':'running','library_sha256':digest,'build_manifest_sha256':sha(build_path.read_bytes()),'scope':'Selected embedding rows and lifecycle only; not full-model/HTTP/performance admission.'}
(out/'workflow.json').write_text(json.dumps(workflow,indent=2),encoding='utf-8')
code=run([sys.executable,str(methods/'host_embedding_fixture.py'),str(root/'models/Qwen3.8-27B-NVFP4'),str(fixture)],out/'fixture-ledger','correctness','prepare-independent-checkpoint-embedding-rows',[methods/'host_embedding_fixture.py',build_path,Path(__file__)])
if not code:
    launcher=root/'test-host-embedding-rows.bat'
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(fixture),str(native),digest],out/'native-ledger','correctness','host-versus-device-embedding-byte-and-lifecycle-gate',[library,launcher,methods/'probe_host_embedding_rows.py',fixture/'fixture.json',build_path,Path(__file__)]+[root/'Garnet'/name for name in build['candidate_source_sha256']])
workflow.update(status='process_failed' if code else 'completed_needs_independent_replay',exit_code=code)
assert sha(library.read_bytes())==digest
for name,want in build['candidate_source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==want
(out/'workflow.json').write_text(json.dumps(workflow,indent=2),encoding='utf-8')
print(json.dumps(workflow,indent=2));raise SystemExit(code)
