"""Prospective fixed 2s conditioning/10x5 ABBA; optional null-only calibration, all original shapes."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('run_vector_vision_context.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns);source=ns['source']
source=source.replace('from make_vector_vision_context import generate','from make_calibrated_vector_vision_context import generate')
source=source.replace("s/'make_vector_vision_context.py',", "s/'make_calibrated_vector_vision_context.py',s/'run_vector_vision_context.py',s/'make_vector_vision_context.py',")
needle='for p in paths:';assert source.count(needle)==1
source=source.replace(needle,"""metadata['selected_variants']=[0] if '--null-only' in sys.argv[3:] else [0,1,2]
metadata['timing_design']=dict(warmup_seconds=2.0,abba_rounds=10,samples_per_window=5,common_alternating_warmup=True,original_numerical_gates=True)
"""+needle)
needle="for variant,label in enumerate(metadata['variants']):";assert source.count(needle)==1
source=source.replace(needle,"for variant in metadata['selected_variants']:\n    label=metadata['variants'][variant]")
for old,new in [('len(windows)==8','len(windows)==40'),('for rnd in (0,1):','for rnd in range(10):'),('len(v[k])==20','len(v[k])==5')]:
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
source=repr(__doc__)+source[source.index('\n'):];ast.parse(source)
destination=Path(sys.argv[2]).with_suffix('.calibrated.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
