"""Preserve terminal retained-zero-source component evidence and exact generator."""
from pathlib import Path
source=Path(__file__).with_name('archive_joint_state_zero_probe.py').read_text().replace('joint-state-zero-*20261015','reused-zero-source-*20261015').replace('joint_state_zero_gpu','reused_zero_source_gpu')
exec(compile(source,__file__,'exec'))
