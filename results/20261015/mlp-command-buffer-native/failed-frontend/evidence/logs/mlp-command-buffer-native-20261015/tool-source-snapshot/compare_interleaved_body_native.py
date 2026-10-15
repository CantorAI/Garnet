"""Native ABBA: identical validated latency bundle, large body layout only."""
from pathlib import Path
import json,sys
wrapper=Path(__file__).with_name('compare_wide_batch_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-wide-batch-prefill','build-interleaved-bf16-body').replace('build-interleaved-screen-index','build-wide-batch-prefill')
source=source.replace('wide-batch-prefill-checks-20261015/jobs/metadata.json','interleaved-body-native-checks-20261015/jobs/metadata.json')
needle="assert metadata['library_sha256']['control']=='837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471'";assert source.count(needle)==1
source=source.replace(needle,"assert metadata['library_sha256']['control']==json.loads((r/'work/wide-batch-next-comparator-selection.json').read_text())['builds']['wide_batch']['sha256']")
source=source.replace("if vl and batch==2 and policy=='candidate':case['chunk']=80","if vl and batch==2:case['chunk']=80")
source=source.replace("expected_plan=([80] if policy=='candidate' else [32,32,16]) if batch==2 else [80]","expected_plan=[80]")
source=source.replace("(wide_count>0)==(vl and batch==2 and policy=='candidate')","(wide_count>0)==(vl and batch==2)")
needle="                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)";assert source.count(needle)==1
source=source.replace(needle,'''                assert value['interleaved_body_enabled']==(policy=='candidate')
                body=value['interleaved_body_counts'];assert len(body)==value['decode_steps_after_first']+1 and len(body)>1
                assert body[0]==0 and all((v>0)==(policy=='candidate') for v in body[1:])
                assert len(set(body[1:]))==1
'''+needle)
source=source.replace("metadata['wide_batch_prefill']=dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[32,32,16]","metadata['wide_batch_prefill']=dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[80]")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
