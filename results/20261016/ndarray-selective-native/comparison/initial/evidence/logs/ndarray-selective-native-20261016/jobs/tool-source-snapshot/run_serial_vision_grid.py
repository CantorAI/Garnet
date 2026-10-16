"""Observe two fresh trained encodes, exact values, admitted/fallback counts and close."""
from pathlib import Path
source=Path(__file__).with_name('run_parallel_vision_grid.py').read_text().replace('parallel_vision_context','serial_vision_prefetch').replace('vision_parallel_dispatches','vision_prefetch_dispatches')
source=source.replace("counts==([f['config']['depth']]*2 if enabled else [0,0])","counts==([f['expected_prefetch_dispatches']]*2 if enabled else [0,0])")
exec(compile(source,__file__,'exec'))
