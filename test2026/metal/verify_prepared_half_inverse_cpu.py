"""Verify captured exhaustive Windows CPU inverse provenance and counts; do not infer Mac behavior."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['jobs']['windows-exhaustive-cpu']['exit_code']==0 and json.loads((root/'exit.json').read_text())==proof['jobs']['windows-exhaustive-cpu']
meta=json.loads((root/'metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert digest(root/'source-snapshot'/Path(name).name)==sha
assert digest(root/'source-snapshot/exact_half_codec.h')=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
assert meta['inverse_branch']=='Portable highest-bit loop: MSVC; Apple clang builtin branch not executed here'
assert meta['architecture']=='AMD64' and 'Windows' in meta['platform'] and meta['full_goal_met'] is False
assert digest(root/'compile.cmd')==meta['compile_script_sha256'] and '/fp:precise' in (root/'compile.cmd').read_text() and '/std:c++17' in (root/'compile.cmd').read_text()
assert proof['omitted']['compiler/cl.exe']['sha256']==meta['compiler_sha256'] and proof['omitted']['prepared_half_inverse_cpu.exe']['sha256']==meta['binary_sha256'] and proof['omitted']['prepared_half_inverse_cpu.obj']['sha256']==meta['object_sha256']
stages=json.loads((root/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile','exhaustive'] and all(s['exit_code']==0 for s in stages)
assert 'Microsoft' in (root/'compile.log').read_text() and 'Compiler' in (root/'compile.log').read_text()
records=[json.loads(line) for line in (root/'exhaustive.log').read_text().splitlines()];assert records==json.loads((root/'report.json').read_text())['records'] and len(records)==1
expected=dict(kind='complete',bf16_scale_cases=8454144,original_codec_admitted=1122816,original_codec_rejected=7331328,half_scale_cases=8454144,finite_half_exact=8189952,nonfinite_half_rejected=264192,invalid_scale_cases=393216,original_codec_and_float_reference_bits_exact=True,rejected_outputs_unchanged=True,gpu_or_model_test=False)
assert records[0]==expected and not json.loads((root/'report.json').read_text())['full_goal_met']
result=dict(verified_files=len(proof['files']),counts=expected,integer_header_sha256=digest(root/'source-snapshot/prepared_half_inverse_bits.h'),scope='Captured exhaustive Windows CPU provenance only; Apple clang CLZ branch, learned packed weights, Mac setup cost, GPU/full-native correctness and speed remain untested.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
