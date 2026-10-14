"""Audit complete requests, actual partial submissions, flags and original gates."""
from pathlib import Path
ns={'__file__':__file__};wrapper=Path(__file__).with_name('verify_vision_global_alpha_native.py').read_text();exec(compile(wrapper[:wrapper.index("if '--confirm'")],__file__,'exec'),ns);source=ns['source']
source=source.replace('vision-global-alpha-staging','prefill-stream-staging').replace('vision-global-alpha-original-source-20261014','prefill-stream-original-source-20261014').replace('vision-global-alpha-checks-20261014','prefill-stream-checks-20261015').replace('vision-global-alpha-native-20261014','prefill-stream-native-20261015').replace('METAL_VISION_GLOBAL_ALPHA','METAL_PREFILL_STREAM')
source=source.replace('Vision global alpha tiny/full rows, FP64, decoder limits, fallback and abort PASS','Prefill stream exact dependencies, ownership, abort drain and recovery PASS 27')
source=source.replace("assert r['vision_global_alpha_enabled']==candidate", "assert not r['vision_global_alpha_enabled'] and not r['dirty_prefix_reset_enabled'] and r['prefill_stream_enabled']==candidate")
needle="assert r['vision_global_alpha_dispatches']==(48 if candidate and p['model']=='Qwen3.5-0.8B' else 0)";assert source.count(needle)==1
source=source.replace(needle,"assert r['vision_global_alpha_dispatches']==0 and r['decode_partial_submissions']==0\n  assert r['prefill_profile'].get('prefill_partial_submissions',0)==(((layers-1)//4)*sum(chunk>1 for chunk in r['prefill_chunk_plan']) if candidate else 0)")
source=source.replace('all_actual_vision_global_alpha_counts_checked','all_actual_prefill_partial_submissions_checked').replace('Vision block global alpha versus global alpha; no fresh vLLM comparison','Ordered prefill submissions versus global alpha; no fresh vLLM comparison')
source=source.replace("actual_vision_global_alpha_counts={policy:[r['vision_global_alpha_dispatches']", "actual_prefill_partial_submissions={policy:[r['prefill_profile'].get('prefill_partial_submissions',0)")
exec(compile(source,__file__,'exec'))
