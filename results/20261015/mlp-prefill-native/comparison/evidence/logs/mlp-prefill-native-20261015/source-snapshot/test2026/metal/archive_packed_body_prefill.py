"""Archive terminal private packed-prefill runs and recheck checkpoint identities."""
from pathlib import Path
source=Path(__file__).with_name('archive_interleaved_bf16_body.py').read_text()
source=source.replace("['interleaved-bf16-body-20261015']","['packed-body-prefill-20261015']")
exec(compile(source,__file__,'exec'))
