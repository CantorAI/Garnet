"""One full unchanged component confirmation after a complete archived/audited initial screen."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent;initial=root/'validation-audits/delta-value-tiles-gpu-guardfix-20261016'

def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

archive=root/'delta-value-tiles-gpu-guardfix-20261016.tar.gz';assert digest(archive)=='59c92a01e6353b28270a06b48d8d801a632d6d5628075bb83e0a9c4bb999c867'
audit=json.loads((initial/'verification.json').read_text());decision=json.loads((initial/'analysis.json').read_text());assert audit['timing_samples']==6400 and audit['numeric_checks']==1872 and decision['unchanged_confirmation_admitted']
selected=[dict(width=w['width'],kind=w['kind']) for w in decision['workloads'] if w['component_screen_pass']];assert selected==[dict(width=64,kind='prefill')]
jobs=initial/'evidence/logs/delta-value-tiles-gpu-guardfix-20261016/jobs';meta=json.loads((jobs/'metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert digest(Path(name))==sha,name
assert digest(root/'models/Qwen3.5-0.8B/model.safetensors')==meta['checkpoint_sha256'] and digest(root/'models/Qwen3.5-0.8B/config.json')==digest(initial/'evidence/config.json')
result=subprocess.check_output([sys.executable,str(tools/'launch_delta_value_tiles_gpu.py'),str(root),str(out)],text=True)
record=dict(initial_archive_sha256=digest(archive),initial_audit_sha256=digest(initial/'verification.json'),initial_analysis_sha256=digest(initial/'analysis.json'),selected_workloads=selected,all_initial_sources_unchanged=True,complete_profiles=8,all_widths_retained=[16,32,64],no_third_favorable_repeat=True,launcher_source_sha256=digest(Path(__file__)),full_goal_met=False)
(out/'confirmation-launch.json').write_text(json.dumps(record,indent=2)+'\n');(out/'confirmation-launcher.py').write_bytes(Path(__file__).read_bytes());print(result.strip())
