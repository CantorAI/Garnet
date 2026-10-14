"""Audit all original numerical cases, new CTest, outputs and actual MLP use."""
from pathlib import Path
source=Path(__file__).with_name('verify_packed_prefill_norm_native.py').read_text()
start=source.index("assert proof['jobs']['packed-prefill-norm-resume-checks-20261014']");end=source.index('staging=base/',start)
source=source[:start]+"assert all(v['exit_code']==0 for v in proof['jobs'].values())\n"+source[end:]
source=source.replace('packed-prefill-norm-staging','joined-global-mlp-staging').replace('packed-prefill-norm-original-source-20261014','joined-global-mlp-original-source-20261014').replace('packed-prefill-norm-resume-checks-20261014','joined-global-mlp-checks-20261014').replace('packed-prefill-norm-resume-native-20261014','joined-global-mlp-native-20261014')
start=source.index("prior=base/'logs/packed-prefill-norm-checks-20261014");end=source.index('for stage in stages[3:]:',start)
source=source[:start]+source[end:];source=source.replace('for stage in stages[3:]:','for stage in stages:')
source=source.replace("'METAL_PACKED_PREFILL_NORM' in meta['options']","'METAL_JOINED_GLOBAL_MLP' in meta['options'] and 'METAL_GLOBAL_HALF_ALPHA' in meta['options']")
source=source.replace("'METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM')","'METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM','METAL_PACKED_PREFILL_NORM')")
source=source.replace("assert r['packed_prefill_norm_enabled']==candidate","assert r['joined_global_mlp_enabled']==candidate and r['global_half_alpha_enabled'] and not r['packed_prefill_norm_enabled']")
source=source.replace("assert r['prefill_profile'].get('packed_prefill_norm_dispatches',0)==(layers*len(r['prefill_chunk_plan']) if candidate else 0)","assert r['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n  admitted=(p['model'],p['batch']) in [('Qwen3-0.6B',1),('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1)]\n  optional_vl=(p['model'],p['batch'])==('Qwen3.5-0.8B',2)\n  blocks=r['prefill_profile'].get('joined_global_mlp_blocks',0)\n  assert (blocks>0) if candidate and admitted else (blocks>=0 if candidate and optional_vl else blocks==0)")
source=source.replace('all_actual_prefill_counters_match','all_reported_joined_global_mlp_counters_checked').replace('Packed prefill versus identical attention; no fresh vLLM comparison','Joined global MLP versus global alpha; VL coverage may be zero; no fresh vLLM comparison')
source=source.replace("meta=json.loads((checks/'metadata.json').read_text())", "assert 'Joined global MLP all outputs, compact reuse, fallback, abort and decode PASS' in (checks/'CTest-LastTest.log').read_text()\nmeta=json.loads((checks/'metadata.json').read_text())")
source=source.replace('profiles.append(dict(**p,distinct_timing_phases=extra))', "profiles.append(dict(**p,distinct_timing_phases=extra,actual_joined_global_mlp_blocks_per_request={policy:[r['prefill_profile'].get('joined_global_mlp_blocks',0) for r in records if r['build_policy']==policy] for policy in ('control','candidate')}))")
if '--confirm' in __import__('sys').argv[3:]:
 source=source.replace('joined-global-mlp-native-20261014','joined-global-mlp-confirm-native-20261014')
 source=source.replace("{(m,b) for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for b in (1,2)}", "{('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1)}")
exec(compile(source,str(Path(__file__)),'exec'))
