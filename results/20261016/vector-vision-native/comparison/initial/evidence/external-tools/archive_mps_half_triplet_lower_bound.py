"""Preserve one terminal synthetic lower-bound success/failure without timing selection."""
from pathlib import Path
import sys
p=Path(__file__).with_name('archive_compact_fused_projection_groups.py');source=p.read_text()
source=source.replace("('compact-fused-projection-groups-20261016','compact-fused-projection-groups-confirm-20261016')","('mps-half-triplet-lower-bound-20261016',)")
source=source.replace('One terminal component cohort; raw outputs/source retained, large binary weights hashed. No full-model gain or independent inference replication.','One terminal synthetic precomputed-plane lower-bound cohort; diagnostics/source retained. No learned model, internal accumulator dtype, or full-model gain claim.')
exec(compile(source,__file__,'exec'))
