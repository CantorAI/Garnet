"""Native ABBA residual-norm candidate versus otherwise identical attention build."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_constant_head.py').read_text()
source=source.replace('build-constant-bf16-head','build-residual-norm').replace('logs/constant-head-checks-20261014','logs/residual-norm-checks-20261014')
source=source.replace("assert value['constant_head_enabled']==(policy=='candidate')","assert value['residual_norm_enabled']==(policy=='candidate')")
source=source.replace("assert value['constant_head_dispatches']==(1 if policy=='candidate' and batch==1 else 0)","assert (value['residual_norm_dispatches']>0) if policy=='candidate' else (value['residual_norm_dispatches']==0)")
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
