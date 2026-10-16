"""Freeze terminal private admission component and original checkpoint identities."""
from pathlib import Path

wrapper=Path(__file__).with_name('archive_interleaved_bf16_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("['interleaved-bf16-body-20261015']","['small-body-threshold-20261015']")
exec(compile(source,__file__,'exec'))
