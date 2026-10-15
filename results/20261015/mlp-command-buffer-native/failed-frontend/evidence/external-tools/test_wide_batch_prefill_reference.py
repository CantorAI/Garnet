"""Persist first/last trained B2 chunk80 layers at the unchanged reference bound."""
from pathlib import Path
source=Path(__file__).with_name('test_replanned_vl_prefill_reference.py').read_text()
assert "for chunk in (16,32,64):" in source
source=source.replace('for chunk in (16,32,64):','for chunk in (80,):')
exec(compile(source,__file__,'exec'))
