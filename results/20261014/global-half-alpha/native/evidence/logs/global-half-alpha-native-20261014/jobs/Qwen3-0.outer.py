"""Native ABBA packed prefill candidate versus identical retained attention."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_constant_head.py').read_text()
source=source.replace('build-constant-bf16-head','build-global-half-alpha').replace('logs/constant-head-checks-20261014','logs/global-half-alpha-checks-20261014')
source=source.replace("assert value['constant_head_enabled']==(policy=='candidate')","assert value['global_half_alpha_enabled']==(policy=='candidate')")
source=source.replace("assert value['constant_head_dispatches']==(1 if policy=='candidate' and batch==1 else 0)","config=json.loads((folder/'config.json').read_text());config=config.get('text_config',config)\\n                assert (value['prefill_profile'].get('global_half_alpha_dispatches',0)>0) if policy=='candidate' else (value['prefill_profile'].get('global_half_alpha_dispatches',0)==0)")
destination=Path(sys.argv[2]).with_suffix('.wrapper.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
