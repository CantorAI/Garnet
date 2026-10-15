"""Reuse frozen27 stage evidence and execute the omitted original wide check."""
from pathlib import Path
import ast,hashlib,json,os,shutil,signal,subprocess,sys,time

r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();s=r/'source/test2026/metal'
prior=r/'logs/mlp-prefill-native-checks-resume-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
archive=r/'mlp-prefill-native-checks-resume-20261015.tar.gz'
with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='61d83b39924e0093104b01850979d1725b9e00afe2ec5875d1f8d4d91cd82153'
manifest=json.loads((r/'mlp-prefill-native-staging/manifest.json').read_text());assert len(manifest)==18
assert all(hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
meta=json.loads((prior/'jobs/metadata.json').read_text());build=r/'build-mlp-prefill-graph-guardfix'
assert hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest()==meta['candidate_sha256']=='23bff0f76a303f2af78c992d4af6e4f90d5041c2e8c57bf454f924b15ac37081'
assert all(hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()==sha for name,sha in meta['source_sha256'].items())
statuses=json.loads((prior/'jobs/statuses.json').read_text());assert len(statuses)==27
assert all(v['exit_code']==0 and v['interruption'] is None for v in statuses)
assert 'wide-batch-trained-reference' not in {v['name'] for v in statuses}
shutil.copytree(prior/'jobs',output)
(output/'original27-statuses.json').write_bytes((prior/'jobs/statuses.json').read_bytes())
(output/'original27-manifest.json').write_bytes((r/'validation-audits/mlp-prefill-native-checks-resume-20261015/evidence/manifest.json').read_bytes())
completion=dict(prior=str(prior),prior_archive_sha256='61d83b39924e0093104b01850979d1725b9e00afe2ec5875d1f8d4d91cd82153',
    reused_stages=[v['name'] for v in statuses],fresh_stages=['wide-batch-trained-reference'],candidate_sha256=meta['candidate_sha256'],source_manifest=manifest)
(output/'completion-provenance.json').write_text(json.dumps(completion,indent=2)+'\n')
background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_guard','exec'),globals())
strict_guard=guard;raw_snapshot=snapshot;owned_group=None
nodes=[n for n in ast.parse((prior/'jobs.executed.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('snapshot','guard','run_once','run')]
assert len(nodes)==4
exec(compile(ast.Module(body=nodes,type_ignores=[]),'preserved_retry_and_ownership','exec'),globals())
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
run('wide-batch-trained-reference',[sys.executable,str(s/'test_wide_batch_prefill_reference.py'),runtime,str(build),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'wide-reference')])
assert len(statuses)==28 and all(v['exit_code']==0 for v in statuses)
assert all(hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
print('All28 original stages accounted for:27 frozen reused,1 fresh wide reference; no accuracy changes',flush=True)
