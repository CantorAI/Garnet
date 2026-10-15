"""Archive complete terminal combined frontend comparisons."""
from pathlib import Path
source=Path(__file__).with_name('archive_striped_frame_inference.py').read_text().replace("glob('striped-frame-inference-*20261015')","glob('threaded-striped-inference-*20261015')")
source=source.replace("names=('frame_preprocessing_striped.py',","names=('frame_threaded_striped.py','frame_webp_threaded.py','run_webp_thread_frame_process.py','run_threaded_striped_frame_process.py','compare_threaded_striped_inference.py','run_threaded_striped_inference.py','launch_threaded_striped_inference.py','frame_preprocessing_striped.py',")
exec(compile(source,__file__,'exec'))
