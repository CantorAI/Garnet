"""Recompute every source-reuse reset gate and the complete balanced windows."""
from pathlib import Path
source=Path(__file__).with_name('verify_joint_state_zero_probe.py').read_text().replace('joint_over_direct_cpu_wall','joint_over_reused_cpu_wall')
needle='        observed.append(r)';assert source.count(needle)==1
source=source.replace(needle,"        assert abs(r['reused_over_legacy_wall']-means['1']['wall_ms']/means['0']['wall_ms'])<1e-12\n"+needle)
exec(compile(source,__file__,'exec'))
