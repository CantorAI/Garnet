"""Preserve all original 19 gates and the exact fused-kernel lifecycle CTest."""
from pathlib import Path
source=Path(__file__).with_name('run_guarded_fused_head_argmax_checks.py').read_text()
source=source.replace('build-fused-head-argmax','build-decode-only-head').replace("'METAL_FUSED_HEAD_ARGMAX'","'METAL_FUSED_HEAD_ARGMAX','METAL_FUSED_HEAD_DECODE_ONLY'")
exec(compile(source,__file__,'exec'))
