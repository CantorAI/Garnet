"""Native ABBA: existing latency bundle versus original-order vision prefetch."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_screened_prefill_reset.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-screened-prefill-reset','build-serial-vision-prefetch').replace('build-screened-head','build-screened-prefill-reset')
source=source.replace('screened-prefill-reset-checks-20261015/jobs/metadata.json','serial-vision-checks-20261015/jobs/metadata.json')
source=source.replace('2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23','75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e')
needle="assert value['prefill_stream_enabled']==(policy=='candidate') and value['dirty_prefix_reset_enabled']==(policy=='candidate')";assert source.count(needle)==1
source=source.replace(needle,"assert value['prefill_stream_enabled'] and value['dirty_prefix_reset_enabled']")
needle=" if policy=='candidate' else 0";assert source.count(needle)==1;source=source.replace(needle,'')
source=source.replace("                if policy=='candidate':\n                    kinds=","                if value['dirty_prefix_reset_enabled']:\n                    kinds=")
needle="                    assert value['frontend_policy']=='combined' and value['frontend_route']=='threaded-striped'";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                    assert value['serial_vision_prefetch_enabled']==(policy=='candidate')
                    expected_prefetch=json.loads((folder/'config.json').read_text())['vision_config']['depth'] if policy=='candidate' else 0
                    assert value['vision_prefetch_dispatches']==expected_prefetch
''')
needle="(output/'metadata.json').write_text(json.dumps(metadata,indent=2))";assert source.count(needle)==1
source=source.replace(needle,"metadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('compare_screened_prefill_reset.py','run_serial_vision_profiles.py')})\n    "+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
