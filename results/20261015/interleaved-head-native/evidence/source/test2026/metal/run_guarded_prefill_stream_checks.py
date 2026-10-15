"""All original nineteen model gates plus the multi-command CTest."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_joined_global_mlp_checks.py').read_text().replace('build-joined-global-mlp','build-prefill-stream').replace('METAL_JOINED_GLOBAL_MLP','METAL_PREFILL_STREAM')
exec(compile(source,__file__,'exec'))
