"""Native six-profile ABBA: same latency/vision bundle, exact seed 8192 vs 1024."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_serial_vision.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-serial-vision-prefetch','build-small-screen-seed'),('build-screened-prefill-reset','build-serial-vision-prefetch'),('serial-vision-checks-20261015/jobs/metadata.json','small-seed-checks-20261015/jobs/metadata.json'),('75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e','b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d')]:source=source.replace(old,new)
source=source.replace("assert value['serial_vision_prefetch_enabled']==(policy=='candidate')","assert value['serial_vision_prefetch_enabled']")
source=source.replace("expected_prefetch=json.loads((folder/'config.json').read_text())['vision_config']['depth'] if policy=='candidate' else 0","expected_prefetch=json.loads((folder/'config.json').read_text())['vision_config']['depth']")
needle="                assert counts==[1]*len(counts)";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                assert value['small_screen_seed_enabled']==(policy=='candidate')\n                assert value['screened_head_seed_rows']==[1024 if policy=='candidate' else 8192]*len(counts)")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
