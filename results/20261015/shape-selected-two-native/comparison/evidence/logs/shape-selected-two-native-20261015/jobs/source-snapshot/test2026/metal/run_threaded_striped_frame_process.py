"""Fresh image decoding and exact striped resize, shared by both runtimes."""
from pathlib import Path
source=Path(__file__).with_name('run_webp_thread_frame_process.py').read_text()
ns={'__file__':__file__}
exec(compile(source[:source.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace("'threaded'","'combined'")
source=source.replace('from frame_webp_threaded import WebPDecoder','from frame_threaded_striped import ThreadedStripedDecoder as WebPDecoder')
needle="            assert list(resized)==metadata['resized_hw']"
assert source.count(needle)==1
source=source.replace(needle,"            assert decoder is None or (decoder.last_decode_route=='native-threaded' and decoder.striped.last_route=='parallel')\n"+needle)
source=source.replace('frontend_setup_seconds=frontend_setup_seconds,',"frontend_setup_seconds=frontend_setup_seconds,frontend_route='threaded-striped' if decoder is not None else 'selected',")
source=source.replace('        worker_log.close()', '        worker_log.close()\n        if decoder is not None:decoder.close()')
exec(compile(source,str(Path(__file__).with_suffix('.executed.py')),'exec'))
