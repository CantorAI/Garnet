"""Matched selected versus combined CPU frontend on unchanged GPU libraries."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_striped_frame_inference.py').read_text()
ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns)
source=ns['source'].replace('run_striped_frame_process.py','run_threaded_striped_frame_process.py').replace("'striped'","'combined'").replace("'parallel'","'threaded-striped'")
source=source.replace('Exact-coordinate striped resize','Combined threaded decode and exact-coordinate striped resize')
needle="metadata['source_sha256'].update("
assert source.count(needle)==1
source=source.replace(needle,"metadata['threaded_decoder_library_sha256']=digest(r/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib')\nassert metadata['threaded_decoder_library_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'\nmetadata['source_sha256'].update({n:digest(s/n) for n in ('frame_threaded_striped.py','frame_webp_threaded.py','run_webp_thread_frame_process.py')})\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
