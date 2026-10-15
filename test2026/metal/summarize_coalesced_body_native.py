"""Join independently audited native block trials; never infer a stale vLLM ratio."""
from pathlib import Path

source = Path(__file__).with_name('summarize_interleaved_body_native.py').read_text()
source = source.replace('interleaved-body-native', 'coalesced-body-native')
source = source.replace("'wide_batch_prefill'):", "'wide_batch_prefill','coalesced_body'):")
source = source.replace(
    "Retain optional text decode candidate; coder control remains preferred. VL decode gain is small and costs about 1 GB; no broad default promotion.",
    "Evaluate repeated decode gains together with TTFT, initialization and extra process memory. No broad promotion or fresh vLLM ratio is established by this join.")
source = source.replace(
    "assert all(a['full_output_requests']==72 and not a['full_goal_met'] for a in audits)",
    "assert all(a['full_output_requests']==72 and a['actual_coalesced'] and a['coalesced_threads']==64 and not a['full_goal_met'] for a in audits)")
exec(compile(source, __file__, 'exec'))
