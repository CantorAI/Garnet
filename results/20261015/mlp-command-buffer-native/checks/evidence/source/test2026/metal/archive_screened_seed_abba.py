"""Preserve terminal same-process screened-seed policies and raw observations."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['screened-seed-abba-20261015']")
exec(compile(source,__file__,'exec'))
