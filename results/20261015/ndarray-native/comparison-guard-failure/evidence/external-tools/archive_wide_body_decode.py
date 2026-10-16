"""Freeze the terminal eight-output component and recheck original checkpoint bytes."""
from pathlib import Path
source=Path(__file__).with_name('archive_interleaved_bf16_body.py').read_text()
source=source.replace("['interleaved-bf16-body-20261015']","['wide-body-decode-20261015']")
exec(compile(source,__file__,'exec'))
