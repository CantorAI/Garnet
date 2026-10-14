"""Verify all full warmed gates and recompute row-specialization component ratios."""
from pathlib import Path
source=Path(__file__).with_name('summarize_constant_bf16.py').read_text()
start=source.index("assert proof['jobs']['constant-bf16-probe-20261014']");end=source.index("jobs=base/'logs'/name/'jobs'",start)
source=source[:start]+"name=sys.argv[3] if len(sys.argv)>3 else 'single-row-bf16-probe-20261014'\nassert proof['jobs'][name]['exit_code']==0\n"+source[end:]
source=source.replace('constant_bf16_gpu','single_row_bf16_gpu')
source=source.replace('Device-readonly/constant BF16 projection components; failed shader-compile attempt discarded','Single-row BF16 components; batch two retains original kernel')
needle=' lookup={c[\'name\']:c for c in assets[\'cases\']}'
assert source.count(needle)==1
source=source.replace(needle,needle+"\n for c in assets['cases']:\n  proof_matrix=proof['omitted_matrices'][model+':'+c['name']];assert proof_matrix['sha256']==c['sha256'] and proof_matrix['bytes']==c['columns']*c['inner']*2 and proof_matrix['checkpoint_sha256']==assets['checkpoint_sha256']")
exec(compile(source,str(Path(__file__)),'exec'))
