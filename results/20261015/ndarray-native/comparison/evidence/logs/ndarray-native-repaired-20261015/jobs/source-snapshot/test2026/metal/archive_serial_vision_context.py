"""Preserve terminal original-order vision components and exact generated code."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['serial-vision-context-20261015']")
exec(compile(source,__file__,'exec'))
