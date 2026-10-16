"""Whole-model ABBA with exact first-head fallback and every decode-use count."""
from pathlib import Path
ns={'__file__':__file__}
wrapper=Path(__file__).with_name('compare_fused_head_argmax.py').read_text()
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace('build-fused-head-argmax','build-decode-only-head').replace('fused-head-argmax-checks-20261015/jobs/metadata.json','decode-only-head-checks-20261015/jobs/metadata.json')
needle="counts=value['fused_head_argmax_counts'];assert len(counts)==value['decode_steps_after_first']+1 and all(count==(1 if policy=='candidate' else 0) for count in counts)"
assert source.count(needle)==1
source=source.replace(needle,"counts=value['fused_head_argmax_counts'];assert len(counts)==value['decode_steps_after_first']+1\n                assert value['fused_head_decode_only_enabled']==(policy=='candidate') and all(chunk>1 for chunk in value['prefill_chunk_plan'])\n                assert counts==([0]+[1]*value['decode_steps_after_first'] if policy=='candidate' else [0]*len(counts))")
exec(compile(source,__file__,'exec'))
