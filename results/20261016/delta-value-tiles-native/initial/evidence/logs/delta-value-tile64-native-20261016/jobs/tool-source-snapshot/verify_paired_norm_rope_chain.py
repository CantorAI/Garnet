"""Reuse every learned bit/clock gate while verifying all 32 carried stages and generated sources."""
from pathlib import Path
import ast
p=Path(__file__).with_name('verify_paired_norm_rope_learned.py');source=p.read_text()
def replace(old,new,count=1):
 global source
 assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)
replace("assert meta['reference_library_sha256']",'''chain_factory=runpy.run_path(str(snapshots['make_paired_norm_rope_chain.py']))
cpp=jobs/'generated/paired_norm_rope_chain.mm'
assert cpp.read_bytes()==chain_factory['probe'](snapshots['paired_norm_rope_learned.mm'].read_text()).encode() and digest(cpp)==meta['generated_sha256'][cpp.name]
expanded=runpy.run_path(str(snapshots['run_paired_norm_rope_chain.py']))['expand'](snapshots['run_paired_norm_rope_learned.py'].read_text())
assert (jobs.parent/'jobs.chain.executed.py').read_bytes()==expanded.encode()
assert meta['reference_library_sha256']''')
replace('dict(common_warmup_ms=2000','dict(chain_stages=32,carried_outputs=True,independent_outputs_per_stage=True,all_stage_outputs_checked=True,common_warmup_ms=2000')
replace('jobs/paired_norm_rope_learned_probe','jobs/paired_norm_rope_chain_probe')
replace("v['original_output_bits_checked']==v['rows']*(c['query_heads']+c['key_heads'])*c['dim']*4", "v['chain_stages']==v['candidate_dispatches_per_execution']==32 and v['original_output_bits_checked']==v['rows']*(c['query_heads']+c['key_heads'])*c['dim']*4*32")
replace("for field in ('input_sha256','output_sha256'):assert len(v[field])==4 and all(len(pair)==2 and all(len(h)==64 for h in pair) for pair in v[field])", """assert len(v['input_sha256'])==4 and all(len(pair)==2 and all(len(h)==64 for h in pair) for pair in v['input_sha256'])
  assert len(v['stage_output_sha256'])==4 and all(len(stages)==32 and all(len(pair)==2 and all(len(h)==64 for h in pair) for pair in stages) for stages in v['stage_output_sha256'])""")
replace("all(v['original_bits_equal'] for v in done)","all(v['original_bits_equal'] and v['chain_stages']==32 for v in done)")
replace("v['complete_output_bits_checked_after_window']", "v['all_stage_output_bits_checked_after_window'] and v['chain_stages']==32 and v['paired_dispatches_per_execution']==(32 if v['candidate'] else 0)")
replace('source_context_regenerated=True,complete_learned_profiles=', 'source_context_regenerated=True,chain_harness_and_runner_regenerated=True,carried_stages=32,complete_learned_profiles=')
replace('All learned component bit/source/clock coverage only. Kernel-only waits and host fixtures are not complete-model inference; no native/matched-vLLM speed proof.', 'All 32 carried stages checked in a distinct learned component calibration. No changed normalization arithmetic; no complete-model or matched-vLLM proof.')
ast.parse(source);exec(compile(source,__file__,'exec'))
