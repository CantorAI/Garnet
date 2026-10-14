"""Recompute exact route counters, all original gates and warm/first clocks."""
from pathlib import Path
source=Path(__file__).with_name('verify_fused_head_argmax_inference.py').read_text()
for old,new in (
    ('fused-head-argmax-checks-20261015','decode-only-head-checks-20261015'),
    ('fused-head-argmax-native-20261015','decode-only-head-native-20261015'),
    ('fused-head-argmax-original-source-20261015','decode-only-head-original-source-20261015'),
    ('build-fused-head-argmax','build-decode-only-head'),
    ("dependencies['pid']==68935","dependencies['pid']==96768")):
    source=source.replace(old,new)
needle="counts=row['fused_head_argmax_counts'];assert len(counts)==row['decode_steps_after_first']+1 and all(v==(1 if policy=='candidate' else 0) for v in counts)"
assert source.count(needle)==1
source=source.replace(needle,"counts=row['fused_head_argmax_counts'];assert len(counts)==row['decode_steps_after_first']+1\n            assert row['fused_head_decode_only_enabled']==(policy=='candidate') and all(chunk>1 for chunk in row['prefill_chunk_plan'])\n            assert counts==([0]+[1]*row['decode_steps_after_first'] if policy=='candidate' else [0]*len(counts))")
source=source.replace("assert 'METAL_FUSED_HEAD_ARGMAX' in meta['options']", "assert 'METAL_FUSED_HEAD_DECODE_ONLY' in meta['options']\nassert 'METAL_FUSED_HEAD_ARGMAX' in meta['options']")
exec(compile(source,__file__,'exec'))
