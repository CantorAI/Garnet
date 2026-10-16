"""Require the complete frozen audit; snapshot both serving and external drivers."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
audit_path=root/'command-buffer-native-checks-verification.json'
archive=root/'command-buffer-native-checks-20261015.tar.gz'
audit=json.loads(audit_path.read_text())
assert audit['validation_stages']==28 and audit['ctests']==8
assert audit['graph_primitive_cases']==audit['dependent_graph_chain_cases']==64
assert audit['screened_recovery_cases']==audit['prefix_recovery_cases']==6
assert audit['wide_reference_profiles']==4 and audit['wide_carried_steps']==12
assert digest(archive)==audit['archive_sha256']
assert digest(root/'build-mlp-command-buffer/libgarnet.dylib')==audit['candidate_library_sha256']
assert digest(root/'build-coalesced-bf16-body/libgarnet.dylib')=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
for name,sha in audit['source_manifest'].items():assert digest(root/'source'/name)==sha,name
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=worker.pid)))
    print(json.dumps(dict(pid=worker.pid,output=str(out))))
else:
    start=time.monotonic();code=1
    try:
        (out/'checks-verification.json').write_bytes(audit_path.read_bytes())
        paths={root/'source'/name for name in audit['source_manifest']}
        identities={p.relative_to(root/'source').as_posix():digest(p) for p in paths}
        for p in paths:
            dest=out/'source-snapshot'/p.relative_to(root/'source');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        (out/'source-identities.json').write_bytes((json.dumps(identities,indent=2)+'\n').encode())
        tools=Path(__file__).resolve().parent
        external={p.name:digest(p) for p in tools.iterdir() if p.is_file() and p.suffix in ('.py','.h','.cpp','.mm','.metal')}
        for name in external:
            dest=out/'tool-source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((tools/name).read_bytes())
        (out/'external-tool-identities.json').write_bytes((json.dumps(dict(directory=str(tools),files=external),indent=2)+'\n').encode())
        code=subprocess.call([sys.executable,str(tools/'run_command_buffer_native_profiles.py'),str(root),str(out/'native')])
        assert all(digest(root/'source'/name)==sha for name,sha in identities.items()),'Serving source changed during native comparison'
        assert all(digest(tools/name)==sha for name,sha in external.items()),'External driver changed during native comparison'
    except Exception:
        code=1;raise
    finally:
        (out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start))+'\n').encode())
    sys.exit(code)
