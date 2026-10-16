"""Full native ABBA; only VL B2 replaces [32,32,16] with a single chunk80."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_interleaved_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=repr(__doc__)+source[source.index('\n'):]
source=source.replace('build-interleaved-screen-index','build-wide-batch-prefill').replace('build-small-screen-seed','build-interleaved-screen-index')
source=source.replace('interleaved-head-checks-20261015/jobs/metadata.json','wide-batch-prefill-checks-20261015/jobs/metadata.json')
source=source.replace('d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876','837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471')
source=source.replace("assert value['interleaved_screen_enabled']==(policy=='candidate')","assert value['interleaved_screen_enabled']")
source=source.replace("value['interleaved_screen_counts']==[1 if policy=='candidate' else 0]*len(counts)","value['interleaved_screen_counts']==[1]*len(counts)")
needle="        for policy,repeat in [('control',0),('candidate',0),('candidate',1),('control',1)]:";assert source.count(needle)==1
source=source.replace(needle,"        original_case=dict(case)\n"+needle+"\n            case=dict(original_case)\n            if vl and batch==2 and policy=='candidate':case['chunk']=80\n            casefile=output/(label+'-'+policy+'-case.json');casefile.write_bytes((json.dumps(case,indent=2)+'\\n').encode())")
needle="                assert value['case']==case";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                assert {k:v for k,v in case.items() if k!='chunk'}=={k:v for k,v in original_case.items() if k!='chunk'}\n                if vl:\n                    expected_plan=([80] if policy=='candidate' else [32,32,16]) if batch==2 else [80]\n                    assert value['prefill_chunk_plan']==expected_plan\n                wide_count=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)\n                assert (wide_count>0)==(vl and batch==2 and policy=='candidate')")
needle="    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))";assert source.count(needle)==1
source=source.replace(needle,"    metadata['wide_batch_prefill']=dict(candidate_vl_b2_plan=[80],control_vl_b2_plan=[32,32,16],logical_inputs_unchanged=True,physical_total_slots=80,no_fresh_vllm_ratio=True)\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
