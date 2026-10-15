"""Exclusive CPU decode+resize composition; all frames freshly read and decoded."""
from pathlib import Path
source=Path(__file__).with_name('run_compact_frame_probe.py').read_text()
source=source.replace('from frame_preprocessing_compact import prepare_frame as candidate', '''from frame_threaded_striped import ThreadedStripedDecoder
library=Path(sys.argv[1])/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib'
assert hashlib.sha256(library.read_bytes()).hexdigest()=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
decoder=ThreadedStripedDecoder(library)''')
source=source.replace("command=[sys.executable,str(s/'test_compact_frame_preprocessing.py'),str(out/'compatibility')]", "command=[sys.executable,str(s/'test_threaded_striped_exact.py'),str(root),str(out/'compatibility'),str(library)]")
needle='  with Image.open(image) as opened:patches,rgb,hw=(candidate if policy else control)(opened,cfg,processor,batch)';assert source.count(needle)==1
source=source.replace(needle,'''  if policy:
   patches,rgb,hw=decoder.prepare(image,cfg,processor,batch,1)
   assert decoder.last_decode_route=='native-threaded' and decoder.striped.last_route=='parallel'
  else:
   with Image.open(image) as opened:patches,rgb,hw=control(opened,cfg,processor,batch)''')
source=source.replace('return dict(batch=batch,candidate=policy,',"return dict(decode_route=decoder.last_decode_route if policy else 'pillow',resize_route=decoder.striped.last_route if policy else 'selected',batch=batch,candidate=policy,")
source=source.replace("paths=[Path(__file__),s/'frame_preprocessing.py',s/'frame_preprocessing_compact.py',s/'test_compact_frame_preprocessing.py',s/'compare_vllm_metal.py']", "paths=[Path(__file__),s/'run_compact_frame_probe.py',s/'frame_preprocessing.py',s/'frame_threaded_striped.py',s/'frame_preprocessing_striped.py',s/'frame_webp_threaded.py',s/'frame_frontend_candidates.py',s/'test_threaded_striped_exact.py',s/'test_webp_thread_exact.py',s/'compare_vllm_metal.py']")
source=source.replace('summary=dict(scope=__doc__,',"summary=dict(library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),scope=__doc__,")
source=source.replace("(out/'summary.json').write_bytes", "decoder.close()\n(out/'summary.json').write_bytes")
exec(compile(source,__file__,'exec'))
