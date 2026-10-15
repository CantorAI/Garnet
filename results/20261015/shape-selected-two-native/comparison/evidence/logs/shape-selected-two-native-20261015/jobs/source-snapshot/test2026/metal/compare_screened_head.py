"""Six unchanged native profiles; actual head counts and identical combined frontend."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_parallel_vision.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-parallel-vision-context','build-screened-head').replace('parallel-vision-checks-20261015/jobs/metadata.json','screened-head-checks-20261015/jobs/metadata.json')
source=source.replace("assert value['parallel_vision_context_enabled']==(policy=='candidate')","assert not value['parallel_vision_context_enabled']")
source=source.replace("assert value['vision_parallel_dispatches']==(json.loads((folder/'config.json').read_text())['vision_config']['depth'] if policy=='candidate' else 0)","assert value['vision_parallel_dispatches']==0")
needle="                assert value['global_half_alpha_enabled']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                assert value['screened_head_enabled']==(policy=='candidate')
                counts=value['screened_head_counts'];assert len(counts)==value['decode_steps_after_first']+1
                assert counts==[(1 if policy=='candidate' else 0)]*len(counts)
                assert not value['fused_head_argmax_enabled'] and not value['body_shape_enabled'] and not value['deferred_state_zero_enabled'] and not value['reused_zero_source_enabled']
''')
needle="metadata['threaded_decoder_sha256']=hashlib";assert source.count(needle)==1
source=source.replace(needle,"metadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('compare_parallel_vision.py','compare_vision_global_alpha.py','run_screened_head_profiles.py','run_screened_head_session_reference.py')})\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
