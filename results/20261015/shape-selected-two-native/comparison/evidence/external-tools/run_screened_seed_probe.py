"""Exact seed-width exploration on immutable trained queries; component only."""
from pathlib import Path
import sys
seed=int(sys.argv[3]);assert seed in (1024,4096,8192,32768)
wrapper=Path(__file__).with_name('run_guarded_screened_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
old='from make_guarded_screened_head import generate_harness as generate,generate_shader';assert source.count(old)==1
source=source.replace(old,'from make_screened_seed_probe import generate_harness as seed_harness,generate_shader as seed_shader\ngenerate=lambda value:seed_harness(value,'+str(seed)+')\ngenerate_shader=lambda value:seed_shader(value,'+str(seed)+')')
source=source.replace("paths=[native,Path(__file__),","paths=[native,Path(__file__),s/'make_screened_seed_probe.py',s/'run_guarded_screened_head.py',")
source=source.replace("metadata=dict(strict_screen_math=True,","metadata=dict(seed_rows="+str(seed)+",strict_screen_math=True,")
source=source.replace('exact 8192-row seed','exact '+str(seed)+'-row seed')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
