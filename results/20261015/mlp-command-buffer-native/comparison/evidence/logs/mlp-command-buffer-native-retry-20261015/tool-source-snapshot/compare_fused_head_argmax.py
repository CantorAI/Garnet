"""Six matched native ABBA profiles; complete tokens and actual fusion counts."""
from pathlib import Path
source=Path(__file__).with_name('compare_vision_global_alpha.py').read_text()
source=source.replace('build-vision-global-alpha','build-fused-head-argmax').replace('vision-global-alpha-checks-20261014/jobs/metadata.json','fused-head-argmax-checks-20261015/jobs/metadata.json')
source=source.replace("assert value['vision_global_alpha_enabled']==(policy=='candidate')", "assert not value['vision_global_alpha_enabled']\n                assert value['fused_head_argmax_enabled']==(policy=='candidate')\n                counts=value['fused_head_argmax_counts'];assert len(counts)==value['decode_steps_after_first']+1 and all(count==(1 if policy=='candidate' else 0) for count in counts)")
source=source.replace("expected_vision=4*json.loads((folder/'config.json').read_text())['vision_config']['depth'] if vl and policy=='candidate' else 0",'expected_vision=0')
source=source.replace('Native ABBA for native BF16 body projection; same frozen workloads and full outputs.','Native ABBA for fused BF16 greedy vocabulary projection; same frozen workloads and full outputs.')
exec(compile(source,__file__,'exec'))
