"""GPU and wall clocks by batch; include all partial/reduction work."""
from pathlib import Path
source=Path(__file__).with_name('summarize_decode_eight_lane.py').read_text().replace('decode-eight-lane-20261015','decode-splitk-20261015').replace('bf16_decode_s8_t64','bf16_decode_sk2').replace('bf16_decode_s8_t128','bf16_decode_sk4')
exec(compile(source,__file__,'exec'))
