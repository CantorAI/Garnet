"""Require quiet inference resources before the isolated global-alpha validation."""
from pathlib import Path
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('checks',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_guarded_global_half_alpha_checks.py'")
exec(compile(source,str(Path(__file__)),'exec'))
