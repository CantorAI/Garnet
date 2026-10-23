"""Summarize component evidence without extrapolating to model performance."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
output=root/'joined-projection-screen-summary.json';assert not output.exists(),'Preserve summary'
source=root/'joined-projection-screen.jsonl';blob=source.read_bytes()
rows=[json.loads(s) for s in blob.decode().splitlines() if s.startswith('{')]
cases=[r for r in rows if r.get('event')=='joined_projection_case'];assert len(cases)==16
assert {(r['model'],r['m']) for r in cases}=={(size,m) for size in ('2b','4b') for m in (4,8,16,32,64,128,256,512)}
assert json.loads((root/'joined-projection-screen-exit.json').read_text())['exit_code']==0
assert json.loads((root/'joined-projection-memcheck-exit.json').read_text())['exit_code']==0
sanitizer=(root/'joined-projection-memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in sanitizer and 'LEAK SUMMARY: 0 bytes leaked' in sanitizer
results=[]
for r in cases:
    assert r['input_bytes_unchanged'] and r['output_guards_intact'] and r['output_splits_bit_exact']
    assert all(g['max_original_fixed_bound_fraction']<=1 for g in r['math'])
    swept=r['eviction_sweep_graph_median_us'];warm=r['warm_graph_median_us']
    results.append({'model':r['model'],'m':r['m'],'bf16_differences':sum(g['bf16_values_different'] for g in r['math']),
        'separate_swept_graph_us':swept[0],'joined_only_swept_graph_us':swept[1],'joined_split_swept_graph_us':swept[2],
        'joined_split_swept_speed_ratio':swept[0]/swept[2],'joined_split_warm_speed_ratio':warm[0]/warm[2],
        'max_fixed_bound_fraction':max(g['max_original_fixed_bound_fraction'] for g in r['math'])})
decode=[r for r in results if r['m']<=32];assert all(r['bf16_differences']==0 for r in decode)
result={'source':source.name,'source_sha256':hashlib.sha256(blob).hexdigest(),'profiles':results,
    'all16_fixed_fp64_bounds_passed':True,'all16_guards_and_input_bytes_passed':True,
    'unfiltered_memcheck_zero_errors_and_leaks':True,'all8_decode_profiles_bf16_identical_in_this_fixture':True,
    'changed_profiles':[{'model':r['model'],'m':r['m'],'bf16_differences':r['bf16_differences']} for r in results if r['bf16_differences']],
    'initial_production_candidate':'Default-off decode width4 only; not yet implemented. Prefill stays unchanged.',
    'scope':'One synthetic random fixture per shape using unchanged production GEMM; nine rotated graph repetitions. 256MiB eviction sweep is not verified L2 residency. Exact fixture output is not a universal arithmetic guarantee. No model timing, quality, memory, or 2x acceptance.'}
output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','source_sha256')},indent=2))
