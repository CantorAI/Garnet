"""Exclusive deferred-view API diagnostic after archived native view failure."""
from pathlib import Path
import sys
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('view',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_view_probe.py'")
exec(compile(source,__file__,'exec'))
