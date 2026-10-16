"""Archive terminal fixed-dimension trials with canonical matrix identities."""
from pathlib import Path
source=Path(__file__).with_name('archive_single_row_bf16_probe.py').read_text().replace("glob('single-row-bf16-*-20261014')","glob('shape-specialized-bf16-*20261015')")
exec(compile(source,__file__,'exec'))
