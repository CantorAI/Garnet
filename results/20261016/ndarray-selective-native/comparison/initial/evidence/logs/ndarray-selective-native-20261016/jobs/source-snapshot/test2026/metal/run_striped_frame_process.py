"""Observed fresh-frame protocol, selected versus exact-coordinate strip resize."""
from pathlib import Path
source=Path(__file__).with_name('run_matched_frame_process.py').read_text()
source=source.replace("('selected','control')","('selected','striped')")
source=source.replace("    prepare=prepare_frame if frontend_policy=='selected' else prepare_frame_control",'''    frontend_setup_begin=time.monotonic()
    striped=None;prepare=prepare_frame
    if frontend_policy=='striped':
        from frame_preprocessing_striped import StripedFramePreparer
        striped=StripedFramePreparer();prepare=striped.prepare
    frontend_setup_seconds=time.monotonic()-frontend_setup_begin''')
needle="            assert list(resized)==metadata['resized_hw']";assert source.count(needle)==1
source=source.replace(needle,"            assert striped is None or striped.last_route=='parallel'\n"+needle)
source=source.replace('                frontend_policy=frontend_policy,',"                frontend_policy=frontend_policy,frontend_setup_seconds=frontend_setup_seconds,frontend_route='parallel' if striped is not None else 'selected',")
source=source.replace('        worker_log.close()', '        worker_log.close()\n        if striped is not None:striped.close()')
exec(compile(source,__file__,'exec'))
