"""Preserve terminal MPS-filter component trials, source and immutable asset hashes."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['mps-screen-index-20261015']")
exec(compile(source,__file__,'exec'))
