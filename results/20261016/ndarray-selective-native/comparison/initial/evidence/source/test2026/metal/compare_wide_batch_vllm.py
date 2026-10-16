"""Fresh matched-FP32 native/vLLM comparison with the validated latency bundle."""
from pathlib import Path
import ast,sys
wrapper=Path(__file__).with_name('compare_screened_head_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
doc=ast.parse(source).body[0];assert isinstance(doc,ast.Expr) and isinstance(doc.value,ast.Constant) and isinstance(doc.value.value,str)
source=repr(__doc__)+'\n'+''.join(source.splitlines(keepends=True)[doc.end_lineno:])
source=source.replace('work/screened-head-next-comparator-selection.json','work/wide-batch-next-comparator-selection.json')
source=source.replace("assert not value['prefill_stream_enabled'] and not value['dirty_prefix_reset_enabled']","assert value['prefill_stream_enabled'] and value['dirty_prefix_reset_enabled']")
source=source.replace('else [80] if batch==1 else [32,32,16]','else [80]')
needle="casefile=output/'case.json';";assert source.count(needle)==1
source=source.replace(needle,"if vl and batch==2:case['chunk']=80\n"+needle)
needle="            assert value['screened_head_enabled']";assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert value['small_screen_seed_enabled'] and value['interleaved_screen_enabled']
            assert value['screened_head_seed_rows']==[1024]*(value['decode_steps_after_first']+1)
            assert value['interleaved_screen_counts']==[1]*(value['decode_steps_after_first']+1)
            wide=value['prefill_profile'].get('wide_batch_prefill_dispatches',0)
            assert (wide>0)==(vl and batch==2)
            if vl:assert value['serial_vision_prefetch_enabled'] and value['vision_prefetch_dispatches']==12
''')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
