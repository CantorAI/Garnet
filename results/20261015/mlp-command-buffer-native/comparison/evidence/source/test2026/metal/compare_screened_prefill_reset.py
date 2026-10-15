"""Native ABBA: screened control versus screened ordered prefill and partial reset."""
from pathlib import Path
import ast,sys
wrapper=Path(__file__).with_name('compare_screened_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-screened-head','build-screened-prefill-reset').replace('build-global-half-alpha','build-screened-head')
source=source.replace('screened-head-checks-20261015/jobs/metadata.json','screened-prefill-reset-checks-20261015/jobs/metadata.json')
source=source.replace('887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f','2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23')
source=source.replace("assert value['screened_head_enabled']==(policy=='candidate')","assert value['screened_head_enabled']").replace("counts==[(1 if policy=='candidate' else 0)]*len(counts)",'counts==[1]*len(counts)')
needle="                expected_vision=0\n                assert value['vision_global_alpha_dispatches']==expected_vision";assert source.count(needle)==1
tree=ast.parse(Path(__file__).with_name('compare_dirty_prefix_reset.py').read_text())
blocks=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "if policy=='candidate':" in n.value and 'dirty_prefix_reset_bytes' in n.value];assert len(blocks)==1
source=source.replace(needle,'''                assert value['prefill_stream_enabled']==(policy=='candidate') and value['dirty_prefix_reset_enabled']==(policy=='candidate')
                assert value['decode_partial_submissions']==0
                submissions=((config['num_hidden_layers']-1)//4)*sum(chunk>1 for chunk in value['prefill_chunk_plan']) if policy=='candidate' else 0
                assert value['prefill_profile'].get('prefill_partial_submissions',0)==submissions
'''+blocks[0].rstrip())
needle="(output/'metadata.json').write_text(json.dumps(metadata,indent=2))";assert source.count(needle)==1
source=source.replace(needle,"metadata['source_sha256'].update({n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('compare_screened_head.py','compare_dirty_prefix_reset.py','run_screened_prefill_reset_profiles.py')})\n    "+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
