"""Exact full native ABBA with real prefill submissions and decode unchanged."""
from pathlib import Path
source=Path(__file__).with_name('compare_vision_global_alpha.py').read_text().replace('build-vision-global-alpha','build-prefill-stream').replace('vision-global-alpha-checks-20261014','prefill-stream-checks-20261015')
source=source.replace("assert value['vision_global_alpha_enabled']==(policy=='candidate')","assert not value['vision_global_alpha_enabled'] and not value['dirty_prefix_reset_enabled']\n                assert value['prefill_stream_enabled']==(policy=='candidate')")
start=source.index('                expected_vision=');end=source.index("                assert not value['single_coder_head_enabled']",start)
source=source[:start]+'''                assert value['vision_global_alpha_dispatches']==0 and value['decode_partial_submissions']==0
                count=((config['num_hidden_layers']-1)//4)*sum(chunk>1 for chunk in value['prefill_chunk_plan']) if policy=='candidate' else 0
                assert value['prefill_profile'].get('prefill_partial_submissions',0)==count
'''+source[end:]
exec(compile(source,__file__,'exec'))
