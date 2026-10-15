"""Fresh-process confirmation of the admitted eight-key original-order variant."""
from pathlib import Path
source=Path(__file__).with_name('run_serial_vision_context.py').read_text()
source=source.replace("paths=[native,Path(__file__),","paths=[native,Path(__file__),s/'run_serial_vision_context.py',")
source=source.replace('metadata=dict(scope=__doc__,','metadata=dict(selected_variants=[4],scope=__doc__,')
source=source.replace("for variant,label in enumerate(metadata['variants']):","for variant,label in [(4,metadata['variants'][4])]:")
exec(compile(source,__file__,'exec'))
