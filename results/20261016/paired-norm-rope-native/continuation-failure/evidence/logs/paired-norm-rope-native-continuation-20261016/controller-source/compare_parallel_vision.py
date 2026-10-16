"""Six native profiles; identical combined frontend on both GPU builds."""
from pathlib import Path
source=Path(__file__).with_name('compare_vision_global_alpha.py').read_text().replace('build-vision-global-alpha','build-parallel-vision-context').replace('vision-global-alpha-checks-20261014/jobs/metadata.json','parallel-vision-checks-20261015/jobs/metadata.json')
source=source.replace('run_matched_frame_process.py','run_threaded_striped_frame_process.py')
needle="str(directory/'protocol')]";assert source.count(needle)==1
source=source.replace(needle,"str(directory/'protocol'),'combined']")
source=source.replace("assert value['vision_global_alpha_enabled']==(policy=='candidate')", "assert not value['vision_global_alpha_enabled']")
source=source.replace("expected_vision=4*json.loads((folder/'config.json').read_text())['vision_config']['depth'] if vl and policy=='candidate' else 0",'expected_vision=0')
needle="                if vl:assert value['vision_reencoded'] and value['patch_sha256']==case['image_metadata']['patch_sha256']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                if vl:
                    assert value['frontend_policy']=='combined' and value['frontend_route']=='threaded-striped'
                    assert value['parallel_vision_context_enabled']==(policy=='candidate')
                    assert value['vision_parallel_dispatches']==(json.loads((folder/'config.json').read_text())['vision_config']['depth'] if policy=='candidate' else 0)
                    assert not value['body_shape_enabled'] and not value['fused_head_argmax_enabled']
                    previous=[v for v in records if v['model_name']==model and v['case']['batch']==batch]
                    assert all(v['rgb_sha256']==value['rgb_sha256'] for v in previous)
''')
needle="metadata=dict(scope=__doc__,";assert source.count(needle)==1
source=source.replace(needle,"metadata=dict(common_frontend='combined threaded decode and exact-coordinate strip resize on both builds',scope=__doc__,")
needle="assert metadata['library_sha256']['control']==";assert source.count(needle)==1
source=source.replace(needle,"metadata['threaded_decoder_sha256']=hashlib.sha256((r/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib').read_bytes()).hexdigest()\nassert metadata['threaded_decoder_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'\nmetadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('frame_threaded_striped.py','frame_webp_threaded.py','frame_preprocessing_striped.py','run_webp_thread_frame_process.py','run_matched_frame_process.py','frame_preprocessing.py')})\n"+needle)
exec(compile(source,__file__,'exec'))
