"""Native ABBA global-alpha prefill versus otherwise identical attention policy."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_packed_prefill_norm.py').read_text()
source=source.replace('build-packed-prefill-norm','build-global-half-alpha').replace('logs/packed-prefill-norm-resume-checks-20261014','logs/global-half-alpha-checks-20261014')
source=source.replace('packed_prefill_norm_enabled','global_half_alpha_enabled')
source=source.replace("assert value['prefill_profile'].get('packed_prefill_norm_dispatches',0)==(config['num_hidden_layers']*len(value['prefill_chunk_plan']) if policy=='candidate' else 0)","assert (value['prefill_profile'].get('global_half_alpha_dispatches',0)>0) if policy=='candidate' else (value['prefill_profile'].get('global_half_alpha_dispatches',0)==0)")
destination=Path(sys.argv[2]).with_suffix('.outer.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
