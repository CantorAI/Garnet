"""Frozen native ABBA: same 1024 seed, original versus interleaved index layout."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_small_seed.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-small-screen-seed','build-interleaved-screen-index'),('build-serial-vision-prefetch','build-small-screen-seed'),('small-seed-checks-20261015/jobs/metadata.json','interleaved-head-checks-20261015/jobs/metadata.json'),('b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d','d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876')]:source=source.replace(old,new)
source=source.replace("assert value['small_screen_seed_enabled']==(policy=='candidate')","assert value['small_screen_seed_enabled']")
source=source.replace("value['screened_head_seed_rows']==[1024 if policy=='candidate' else 8192]*len(counts)","value['screened_head_seed_rows']==[1024]*len(counts)")
needle="                assert counts==[1]*len(counts)";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                assert value['interleaved_screen_enabled']==(policy=='candidate')\n                assert value['interleaved_screen_counts']==[1 if policy=='candidate' else 0]*len(counts)")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
