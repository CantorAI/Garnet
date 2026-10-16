"""Independent original inputs for decode+resize composition and all fallback cases."""
from pathlib import Path
source=Path(__file__).with_name('test_webp_thread_exact.py').read_text()
source=source.replace('from frame_webp_threaded import WebPDecoder','from frame_threaded_striped import ThreadedStripedDecoder as WebPDecoder')
source=source.replace('metadata=dict(checks=checks,',"assert decoder.striped.parallel_calls>0 and decoder.striped.fallback_calls>0\nmetadata=dict(parallel_resize_calls=decoder.striped.parallel_calls,fallback_resize_calls=decoder.striped.fallback_calls,checks=checks,")
source=source.replace("(Path(__file__),s/'frame_webp_threaded.py',s/'frame_preprocessing.py')", "(Path(__file__),s/'test_webp_thread_exact.py',s/'frame_threaded_striped.py',s/'frame_preprocessing_striped.py',s/'frame_webp_threaded.py',s/'frame_preprocessing.py',s/'frame_frontend_candidates.py')")
try:exec(compile(source,__file__,'exec'))
finally:
    if 'decoder' in globals():decoder.close()
