"""Run only after primary observes current reproof tool session terminal exit0."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';opt=root/'work/worldsense-opt';r=Path('/Users/shawn/CantorAI');session=int(sys.argv[1]);assert session==2138
result=json.loads((w/'reproof-complete.json').read_text());assert result['full_native_text_VLM_corpus_vision_lifecycle_and_parent_telemetry_reproof_pass']and result['source_unchanged']
identity=json.loads((w/'reproof-driver-source-library-identity.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in identity['files'].items())
source=r/'out/research/worldsense-native-text-worker/source';assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==sha for k,sha in identity['source'].items())
assert all(hashlib.sha256((r/('out/build/garnet-metal-worldsense-native-text-worker-'+mode)/'libgarnet.dylib').read_bytes()).hexdigest()==sha for mode,sha in identity['libraries'].items())
(w/'reproof-observed-terminal.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session_id=session,observed_terminal_exit=0,all_consumed_driver_source_library_identities_unchanged=True,source_manifest_sha256=hashlib.sha256((w/'gateway-source-manifest.json').read_bytes()).hexdigest(),identity_sha256=hashlib.sha256((w/'reproof-driver-source-library-identity.json').read_bytes()).hexdigest(),scope='Terminal observation and identity audit, plus separate independent numerical verifier. Not performance.'),indent=2)+'\n')
subprocess.run([sys.executable,str(w/'verify_correctness_terminal.py'),str(session)],check=True)
# Prepare subsequent-stage metadata only after preserving all consumed identities.
p=opt/'ws58_postproof.py';t=p.read_text();needle="str(w/'verify_correctness_terminal.py'),'48405'";assert t.count(needle)==1;t=t.replace(needle,"str(w/'verify_correctness_terminal.py'),sys.argv[1]");p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text();needle="['controller_session']==48405";assert t.count(needle)==1;t=t.replace(needle,"['controller_session']==int(json.loads((w/'reproof-observed-terminal.json').read_text())['session_id'])");p.write_text(t)
subprocess.run([sys.executable,str(opt/'ws58_postproof.py'),str(session)],check=True)
subprocess.run([sys.executable,str(opt/'audit_routes_ws58.py')],check=True)
print('Current reproof terminal/identities/full numerical/postfault/config/gate-route audits PASS; benchmark refreeze remains required',flush=True)
