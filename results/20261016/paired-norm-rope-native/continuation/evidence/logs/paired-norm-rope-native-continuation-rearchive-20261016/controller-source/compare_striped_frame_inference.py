"""Same selected GPU model builds; common exact strip resize applied to both runtimes."""
from pathlib import Path
import sys
source=Path(__file__).with_name('compare_frame_frontend_inference.py').read_text().replace('run_matched_frame_process.py','run_striped_frame_process.py')
source=source.replace('Fresh-frame copy elision applied equally to Garnet/vLLM, sequential factorial ABBA.','Exact-coordinate striped resize applied equally to Garnet/vLLM, sequential factorial ABBA.')
source=source.replace("'selected'","'striped'").replace("'control'","'selected'")
source=source.replace("build=r/'build-resident-kv-inplace'","build=r/'build-global-half-alpha'")
source=source.replace('73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f')
source=source.replace('def guard(label):\n    strict_guard()', '''def guard(label):
    import shutil
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()''')
needle="(output/'metadata.json').write_text(json.dumps(metadata,indent=2))";assert source.count(needle)==1
source=source.replace(needle,"metadata['source_sha256'].update({n:digest(s/n) for n in ('frame_preprocessing_striped.py','compare_frame_frontend_inference.py','run_matched_frame_process.py')})\n"+needle)
needle="            assert value['prefill_chunk_plan']==expected_plan";assert source.count(needle)==1
source=source.replace(needle,needle+"\n            assert value['global_half_alpha_enabled'] and not value['body_shape_enabled']\n            assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0")
needle="        assert value['case']==case and value['frontend_policy']==frontend_policy";assert source.count(needle)==1
source=source.replace(needle,needle+"\n        assert value['frontend_route']==('parallel' if frontend_policy=='striped' else 'selected') and value['frontend_setup_seconds']>=0")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
