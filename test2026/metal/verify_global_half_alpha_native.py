"""Audit full global-alpha validation/output/counter evidence and measured phases."""
from pathlib import Path
source=Path(__file__).with_name('verify_packed_prefill_norm_native.py').read_text()
start=source.index("assert proof['jobs']['packed-prefill-norm-resume-checks-20261014']");end=source.index("staging=base/",start)
source=source[:start]+"assert all(v['exit_code']==0 for v in proof['jobs'].values())\n"+source[end:]
source=source.replace('packed-prefill-norm-staging','global-half-alpha-staging').replace('packed-prefill-norm-original-source-20261014','global-half-alpha-original-source-20261014').replace('packed-prefill-norm-resume-checks-20261014','global-half-alpha-checks-20261014').replace('packed-prefill-norm-resume-native-20261014','global-half-alpha-native-20261014')
start=source.index("prior=base/'logs/packed-prefill-norm-checks-20261014");end=source.index('for stage in stages[3:]:',start)
source=source[:start]+source[end:];source=source.replace('for stage in stages[3:]:','for stage in stages:')
source=source.replace("'METAL_PACKED_PREFILL_NORM' in meta['options']","'METAL_GLOBAL_HALF_ALPHA' in meta['options']").replace("'METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM')","'METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM','METAL_PACKED_PREFILL_NORM')")
source=source.replace("r['packed_prefill_norm_enabled']==candidate","r['global_half_alpha_enabled']==candidate")
source=source.replace("assert r['prefill_profile'].get('packed_prefill_norm_dispatches',0)==(layers*len(r['prefill_chunk_plan']) if candidate else 0)","assert (r['prefill_profile'].get('global_half_alpha_dispatches',0)>0) if candidate else (r['prefill_profile'].get('global_half_alpha_dispatches',0)==0)\n  assert not r['packed_prefill_norm_enabled']")
source=source.replace('all_actual_prefill_counters_match','all_actual_global_alpha_counters_match').replace('Packed prefill versus identical attention; no fresh vLLM comparison','Global alpha versus identical attention; no fresh vLLM comparison')
if '--confirm' in __import__('sys').argv[3:]:
 source=source.replace('global-half-alpha-native-20261014','global-half-alpha-confirm-native-20261014')
exec(compile(source,str(Path(__file__)),'exec'))
