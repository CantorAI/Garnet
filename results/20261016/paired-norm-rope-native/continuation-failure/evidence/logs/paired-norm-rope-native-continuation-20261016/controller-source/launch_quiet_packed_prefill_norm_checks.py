"""Require a quiet interval before building/validating the optional prefill path."""
from pathlib import Path
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('checks',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_packed_prefill_norm_checks.py'")
exec(compile(source,str(Path(__file__)),'exec'))
