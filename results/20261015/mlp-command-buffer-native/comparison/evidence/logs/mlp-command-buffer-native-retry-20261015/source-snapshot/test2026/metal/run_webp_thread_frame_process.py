"""Measured frame protocol with selected Pillow versus optional threaded WebP."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_matched_frame_process.py').read_text()
source=source.replace("('selected','control')", "('selected','threaded')")
source=source.replace("    prepare=prepare_frame if frontend_policy=='selected' else prepare_frame_control", '''    frontend_setup_begin=time.monotonic()
    prepare=prepare_frame
    from frame_webp_threaded import WebPDecoder
    root=Path(source).parent
    decoder=WebPDecoder(root/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib') if frontend_policy=='threaded' else None
    frontend_setup_seconds=time.monotonic()-frontend_setup_begin''')
needle="            with Image.open(metadata['image']) as image:patches,rgb,resized=prepare(image,configuration['vision_config'],processor,batch)"
assert source.count(needle)==1
source=source.replace(needle,'''            if decoder is None:
                with Image.open(metadata['image']) as image:patches,rgb,resized=prepare(image,configuration['vision_config'],processor,batch)
            else:patches,rgb,resized=decoder.prepare(metadata['image'],configuration['vision_config'],processor,batch,1)''')
source=source.replace("                frontend_policy=frontend_policy,", "                frontend_policy=frontend_policy,frontend_setup_seconds=frontend_setup_seconds,")
exec(compile(source,str(Path(__file__).with_suffix('.executed.py')),'exec'))
