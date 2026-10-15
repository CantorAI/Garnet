"""Same native library and frozen VL B2 inputs: retune internal chunk32 to16/64."""
from pathlib import Path
import sys
maximum=int(sys.argv[3]);assert maximum in (16,64)
assert len(sys.argv)==4
sys.argv.extend(['--model','Qwen3.5-0.8B','--batch','2'])
wrapper=Path(__file__).with_name('compare_interleaved_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=repr(__doc__)+source[source.index('\n'):]
source=source.replace("('control','build-small-screen-seed')","('control','build-interleaved-screen-index')")
source=source.replace("assert metadata['library_sha256']['control']=='d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876'","assert metadata['library_sha256']['control']==metadata['library_sha256']['candidate']")
source=source.replace("assert value['interleaved_screen_enabled']==(policy=='candidate')","assert value['interleaved_screen_enabled']")
source=source.replace("value['interleaved_screen_counts']==[1 if policy=='candidate' else 0]*len(counts)","value['interleaved_screen_counts']==[1]*len(counts)")
needle="        for policy,repeat in [('control',0),('candidate',0),('candidate',1),('control',1)]:";assert source.count(needle)==1
source=source.replace(needle,"        original_case=dict(case);assert model=='Qwen3.5-0.8B' and batch==2 and original_case['chunk']==32\n"+needle+"\n            case=dict(original_case);case['chunk']=32 if policy=='control' else "+str(maximum)+"\n            casefile=output/(label+'-'+policy+'-case.json');casefile.write_bytes((json.dumps(case,indent=2)+'\\n').encode())")
needle="                assert value['case']==case";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                expected_plan=[32,32,16] if policy=='control' else "+repr([16]*5 if maximum==16 else [64,16])+"\n                assert value['prefill_chunk_plan']==expected_plan\n                assert {k:v for k,v in case.items() if k!='chunk'}=={k:v for k,v in original_case.items() if k!='chunk'}")
needle="    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))";assert source.count(needle)==1
source=source.replace(needle,"    metadata['internal_prefill_replan']=dict(control_maximum=32,candidate_maximum="+str(maximum)+",same_library=True,logical_inputs_unchanged=True,physical_total_slots=80,no_fresh_vllm_ratio=True)\n"+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
