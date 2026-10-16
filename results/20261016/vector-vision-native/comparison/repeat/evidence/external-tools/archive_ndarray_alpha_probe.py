"""Keep alpha-path diagnosis separate from the original failed compatibility gate."""
from pathlib import Path
p=Path(__file__).with_name('archive_ndarray_compat.py');source=p.read_text()
source=source.replace("trial='ndarray-compat-20261015'","trial='ndarray-alpha-compat-20261015'")
exec(compile(source,__file__,'exec'))
