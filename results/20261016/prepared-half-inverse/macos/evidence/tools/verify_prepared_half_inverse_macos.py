"""Verify captured actual Apple compiler branch and exhaustive exact CPU result."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['jobs']['macos-exhaustive-cpu']['exit_code']==0 and json.loads((root/'exit.json').read_text())==proof['jobs']['macos-exhaustive-cpu']
meta=json.loads((root/'metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert digest(root/'source-snapshot'/Path(name).name)==sha
assert digest(root/'source-snapshot/exact_half_codec.h')=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9' and digest(root/'source-snapshot/prepared_half_inverse_bits.h')=='2ba0f45baff48f129743b7ee1c54c941ef8a0ec43e58ed686670a8a72882f7c5'
assert 'Apple clang' in meta['compiler_version'] and 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
macros=(root/'macros.log').read_text();assert all(v in macros for v in ('#define __clang__ 1','#define __APPLE__ 1','#define __aarch64__ 1'))
stages=json.loads((root/'statuses.json').read_text());assert [v['name'] for v in stages]==['macros','compile','exhaustive'] and all(v['exit_code']==0 for v in stages)
assert all(v in stages[1]['command'] for v in ('-std=c++17','-O3','-fno-fast-math','-ffp-contract=off')) and not any(v in stages[1]['command'] for v in ('Metal','MetalPerformanceShaders'))
assert proof['omitted']['compiler/clang++']['sha256']==meta['compiler_sha256'] and proof['omitted']['prepared_half_inverse_cpu']['sha256']==meta['binary_sha256']
expected=dict(kind='complete',bf16_scale_cases=8454144,original_codec_admitted=1122816,original_codec_rejected=7331328,half_scale_cases=8454144,finite_half_exact=8189952,nonfinite_half_rejected=264192,invalid_scale_cases=393216,original_codec_and_float_reference_bits_exact=True,rejected_outputs_unchanged=True,gpu_or_model_test=False)
records=[json.loads(line) for line in (root/'exhaustive.log').read_text().splitlines()];assert records==[expected] and records==json.loads((root/'report.json').read_text())['records']
result=dict(verified_files=len(proof['files']),counts=expected,integer_header_sha256=digest(root/'source-snapshot/prepared_half_inverse_bits.h'),apple_clang_clz_branch=True,scope='Captured Apple-clang exhaustive mathematical CPU verification only. Learned packing, setup performance and any changed GPU/native source still need their own tests.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
