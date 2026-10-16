"""Preserve every original numerical/shape/guard predicate; audit prospective timing design."""
from pathlib import Path
import ast,sys
source=Path(__file__).with_name('verify_vector_vision_context.py').read_text()
source=source.replace('make_vector_vision_context.py','make_calibrated_vector_vision_context.py')
source=source.replace("sys.modules.pop('make_serial_vision_context',None)","sys.modules.pop('make_serial_vision_context',None)\n    sys.modules.pop('make_vector_vision_context',None)")
for old,new in [('len(windows)==8','len(windows)==40'),('for rnd in (0,1):','for rnd in range(10):'),
                ('len(v[k])==20','len(v[k])==5'),('len(profiles)*160','len(profiles)*200')]:
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
needle="    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha";assert source.count(needle)==1
source=source.replace(needle,needle+"\n    assert meta['timing_design']==dict(warmup_seconds=2.0,abba_rounds=10,samples_per_window=5,common_alternating_warmup=True,original_numerical_gates=True)")
ast.parse(source);Path(sys.argv[1]).write_bytes(source.encode())
