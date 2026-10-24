"""Guarded host-embedding native build; never run while original GPU study is live.

Stage exactly the handoff's three candidate files under results/host-embedding-source.
Preserves old source bytes and references the already retained original DLL.
"""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results'
methods=root/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
out=results/'native-host-embedding-build-attempt1';assert not out.exists()
prior=results/'batch-mtp-deferred-http-pilot-attempt1'
events=[json.loads(line) for line in (prior/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
workflow=json.loads((prior/'workflow.json').read_text());assert workflow['exit_code']==0
proof_path=results/'batch-mtp-deferred-http-pilot-attempt1-verified.json'
proof=json.loads(proof_path.read_text())
assert proof['status']=='batch_mtp_deferred_http_pilot_independently_verified'
archive=results/'batch-mtp-deferred-http-pilot-attempt1-evidence.zip'
assert sha(archive.read_bytes())==proof['archive_sha256']
plan_path=methods/'evidence/host-embedding-source-handoff.json';plan=json.loads(plan_path.read_text())
assert plan['status']=='host_embedding_source_handoff_prepared_not_deployed'
names={'src/backends/tensorrt/cuda_tensor_io.cpp','src/backends/tensorrt/cuda_tensor_io.h','xModel/qwen35/vl_0_8b/cuda_packed_decoder.py'}
assert set(plan['original_source_sha256'])==set(plan['candidate_source_sha256'])==names
old_library=plan['original_library_sha256'];library=root/'build/bin/garnet.dll'
retained=root/'build/bin/garnet-native-full-attention-fp8-attempt1.dll'
assert sha(library.read_bytes())==sha(retained.read_bytes())==old_library==proof['library_sha256']
policy_path=results/'native-full-attention-fp8-build-attempt1/manifest.json';policy=json.loads(policy_path.read_text())
assert policy['library_sha256']==old_library
for name,digest in policy['source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==digest
old={name:(root/'Garnet'/name).read_bytes() for name in names}
new={name:(results/'host-embedding-source'/name).read_bytes() for name in names}
for name in names:
    assert sha(old[name])==plan['original_source_sha256'][name],('old source drift',name)
    assert sha(new[name])==plan['candidate_source_sha256'][name],('staged source drift',name)
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
launcher=root/'build-layer0-stages.bat';assert launcher.is_file()
out.mkdir()
for name,raw in old.items():
    path=out/'original'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
receipt={'status':'installing_unbuilt_candidate','old_library_sha256':old_library,'retained_library':str(retained),
    'handoff_sha256':sha(plan_path.read_bytes()),'prior_http_proof_sha256':sha(proof_path.read_bytes()),
    'original_source_sha256':plan['original_source_sha256'],'candidate_source_sha256':plan['candidate_source_sha256'],
    'unchanged_policy_source_sha256':policy['source_sha256']}
(out/'source-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
code=None;failure=None
try:
    for name,raw in new.items():
        (root/'Garnet'/name).write_bytes(raw);assert sha((root/'Garnet'/name).read_bytes())==plan['candidate_source_sha256'][name]
    code=run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','host-embedding-default-off-candidate',
        [launcher,library,plan_path,proof_path,policy_path,Path(__file__)]+[root/'Garnet'/name for name in sorted(names)])
    if code:raise RuntimeError('Native build failed with exit '+str(code))
    assert library.is_file() and sha(library.read_bytes())!=old_library
    for name,digest in policy['source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==digest
    for name,digest in plan['candidate_source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==digest
    saved=root/'build/bin/garnet-native-host-embedding-attempt1.dll';assert not saved.exists();saved.write_bytes(library.read_bytes())
    receipt.update(status='native_host_embedding_build_finished_unqualified',exit_code=0,library_sha256=sha(library.read_bytes()),retained_candidate_library=str(saved))
except BaseException as error:
    failure=repr(error)
    receipt.update(status='native_host_embedding_build_failed_restored',exit_code=code,error=failure,
        failed_output_library_sha256=sha(library.read_bytes()) if library.is_file() else None)
    if library.is_file() and sha(library.read_bytes())!=old_library:
        failed=out/'failed-garnet.dll';assert not failed.exists();failed.write_bytes(library.read_bytes())
        receipt['retained_failed_output_library']=str(failed)
    library.write_bytes(retained.read_bytes());assert sha(library.read_bytes())==old_library
    for name,raw in old.items():(root/'Garnet'/name).write_bytes(raw)
    for name,digest in plan['original_source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==digest
    receipt['restored_original_library_sha256']=sha(library.read_bytes())
finally:
    receipt['scope']='Native build/guarded source handoff only. Not default regression, embedding rows, full-model state/IDs, streaming, sanitizer, memory saving or performance qualification.'
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
if failure:raise SystemExit(code or 1)
