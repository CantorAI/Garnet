"""Freeze the deferred-view API diagnostic before any native repair."""
from pathlib import Path
source=Path(__file__).with_name('archive_ndarray_compat.py').read_text().replace("trial='ndarray-compat-20261015'","trial='ndarray-view-compat-20261015'")
exec(compile(source,__file__,'exec'))
