"""Matched native/vLLM factorial ABBA: identical fresh pixels with two CPU paths."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_frame_frontend_inference.py').read_text()
source=source.replace('run_matched_frame_process.py','run_webp_thread_frame_process.py')
source=source.replace("'selected'","'threaded'").replace("'control'","'selected'")
needle="build=r/'build-resident-kv-inplace'"
assert source.count(needle)==1
source=source.replace(needle,"build=r/('build-attention-prefix' if batch==1 else 'build-resident-kv-inplace')")
needle="assert metadata['native_library_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'"
source=source.replace(needle,"assert metadata['native_library_sha256']==('f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088' if batch==1 else '73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2')")
source=source.replace('def guard(label):\n    strict_guard()', '''def guard(label):
    import shutil
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()''')
needle="(output/'metadata.json').write_text(json.dumps(metadata,indent=2))"
source=source.replace(needle,"metadata.update(frontend_bridge_sha256=digest(r/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib'))\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
