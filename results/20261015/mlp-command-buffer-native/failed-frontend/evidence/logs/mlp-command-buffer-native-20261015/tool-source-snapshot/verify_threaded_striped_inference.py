"""Verify full outputs, exact pixels, actual GPU precision and matched timings."""
from pathlib import Path
source=Path(__file__).with_name('verify_striped_frame_inference.py').read_text().replace("'striped'","'combined'").replace("'parallel'","'threaded-striped'")
needle="        assert meta['native_library_sha256']==proof['native_library_sha256'] and meta['learned_values_bit_exact']"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n        assert meta['threaded_decoder_library_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'")
exec(compile(source,__file__,'exec'))
