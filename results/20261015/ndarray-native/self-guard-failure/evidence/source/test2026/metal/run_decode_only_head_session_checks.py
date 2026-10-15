"""Retain all six trained greedy/fallback/isolation/recovery cases."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_checks.py').read_text().replace('build-fused-head-argmax','build-decode-only-head')
exec(compile(source,__file__,'exec'))
