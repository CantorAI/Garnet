"""Fresh matched-FP32 screened-head versus vLLM with identical image frontend."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_global_alpha_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('work/global-alpha-next-comparator-selection.json','work/screened-head-next-comparator-selection.json')
source=source.replace('run_matched_frame_process.py','run_threaded_striped_frame_process.py')
needle="str(folder/'protocol')]";assert source.count(needle)==1
source=source.replace(needle,"str(folder/'protocol'),'combined']")
needle="            assert value['attention_prefix_enabled']==selected['attention_prefix_enabled']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert value['screened_head_enabled']
            counts=value['screened_head_counts'];assert counts==[1]*(value['decode_steps_after_first']+1)
            assert not value['fused_head_argmax_enabled'] and not value['body_shape_enabled']
            assert not value['prefill_stream_enabled'] and not value['dirty_prefix_reset_enabled']
            assert not value['deferred_state_zero_enabled'] and not value['reused_zero_source_enabled']
''')
needle="            assert value['patch_sha256']==case['image_metadata']['patch_sha256']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert value['frontend_policy']=='combined' and value['frontend_route']=='threaded-striped'
            assert all(v['rgb_sha256']==value['rgb_sha256'] for v in records)
''')
needle="(output/'metadata.json').write_text(json.dumps(metadata,indent=2))";assert source.count(needle)==1
source=source.replace(needle,'''metadata['common_frontend']='combined threaded decode and exact-coordinate strip resize for both runtimes'
metadata['threaded_decoder_sha256']=digest(r/'logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib')
assert metadata['threaded_decoder_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
metadata['source_sha256'].update({n:digest(s/n) for n in ('compare_global_alpha_vllm.py','compare_retained_prefill_vllm.py','run_screened_head_vllm.py','frame_threaded_striped.py','frame_webp_threaded.py','frame_preprocessing_striped.py','run_webp_thread_frame_process.py','run_matched_frame_process.py','frame_preprocessing.py')})
'''+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
