"""Archive terminal seed-width trials with raw logs and original asset hashes."""
from pathlib import Path
source=Path(__file__).with_name('archive_padded_mps_body.py').read_text().replace("['padded-mps-body-20261015']","['screened-seed-sweep-20261015']")
exec(compile(source,__file__,'exec'))
