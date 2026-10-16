"""Require terminal full32 finalizer, exact archived re-audit and exclusive quiet model execution."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent']);tools=Path(__file__).parent
gate=json.loads((root/'delta-value-tile64-native-checks-gate.json').read_text());assert gate['validation_stages']==32
finalizer=Path(gate['finalizer_folder']);assert finalizer.is_relative_to(parent/'logs') and json.loads((finalizer/'exit.json').read_text())['exit_code']==0
pid=int(json.loads((finalizer/'pid.json').read_text())['pid']);state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'))
archive=Path(gate['checks_archive']);extracted=Path(gate['checks_audit_directory']);assert archive.is_relative_to(parent) and extracted.is_relative_to(parent/'validation-audits')

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

assert digest(archive)==gate['archive_sha256'] and digest(extracted/'verification.json')==gate['full_audit_sha256']
with tarfile.open(archive) as tar:proof=json.load(tar.extractfile('evidence/manifest.json'))
assert digest(tools/'verify_delta_value_tiles_native_checks.py')==proof['files']['evidence/external-archive-tools/verify_delta_value_tiles_native_checks.py']['sha256']
destination=root/'delta-value-tile64-native-checks-archive-verification.json'
subprocess.run([sys.executable,str(tools/'verify_delta_value_tiles_native_checks.py'),str(extracted),str(destination)],check=True)
audit=json.loads(destination.read_text());assert audit['validation_stages']==32 and audit['delta_value_tile64_trained_steps']==48 and audit['candidate_library_sha256']==gate['candidate_sha256'];audit['archive_sha256']=gate['archive_sha256'];destination.write_text(json.dumps(audit,indent=2)+'\n')
source=tools.joinpath('launch_quiet_prefill_residual_norm_probe.py').read_text().replace("mode in ('regular','extra')","mode in ('profiles',)").replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_delta_value_tiles_native_profiles.py'").replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
