"""Verify CPU archive identities and reproduce the bounded indexing proof from its frozen scripts."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text());assert proof['exit_code']==0
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
report=json.loads((root/'report.json').read_text())
for name,sha in report['source_sha256'].items():assert digest(root/'source-snapshot'/Path(name).name)==sha
assert len(report['profiles'])==12 and report['original_qk_norm_prefix_bytes_preserved'] and report['per_value_original_arithmetic_body_recovered'] and report['all_state_and_output_words_owned_once']
assert not report['gpu_executed'] and not report['numerical_equivalence_proven'] and not report['performance_measured'] and not report['full_goal_met']
assert digest(root/'delta-value-tiles.metal')==report['generated_shader_sha256']
scratch=(Path.cwd()/'.work').resolve();scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='delta-tile-address-reproduce-',dir=str(scratch)) as temporary:
 target=Path(temporary).resolve();assert target.is_relative_to(scratch)
 command=[sys.executable,'-X','utf8',str(root/'source-snapshot/audit_delta_value_tile_addresses.py'),str(root/'source-snapshot/metal_context.mm'),str(target/'proof')]
 p=subprocess.run(command,capture_output=True,text=True);assert p.returncode==0,p.stderr;repeated=json.loads((target/'proof/report.json').read_text())
 for field in ('profiles','generated_shader_sha256','original_kernel_sha256','original_qk_norm_prefix_bytes_preserved','per_value_original_arithmetic_body_recovered','all_state_and_output_words_owned_once','gpu_executed','numerical_equivalence_proven','performance_measured','scope','full_goal_met'):assert repeated[field]==report[field],field
 assert {Path(k).name:v for k,v in repeated['source_sha256'].items()}=={Path(k).name:v for k,v in report['source_sha256'].items()}
result=dict(verified_files=len(proof['files']),reproduced_cpu_profiles=12,state_words_checked=sum(v['unique_state_words'] for v in report['profiles']),output_words_checked=sum(sum(v['output_words_by_chunk'].values()) for v in report['profiles']),source_and_generator_reproduced=True,gpu_executed=False,numerical_equivalence_proven=False,performance_measured=False,scope='Reproduced CPU generated-address/ownership and literal-body prerequisite only; actual GPU and complete-model gates remain required.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
