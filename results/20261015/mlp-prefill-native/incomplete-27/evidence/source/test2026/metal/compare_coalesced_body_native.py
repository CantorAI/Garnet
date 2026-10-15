"""Native complete-output ABBA: identical latency bundle, exact block decode only."""
from pathlib import Path
wrapper=Path(__file__).with_name('compare_interleaved_body_native.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body')
source=source.replace('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015')
needle="                assert value['interleaved_body_enabled']==(policy=='candidate')";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                assert value['coalesced_block_enabled']==(policy=='candidate')
                compressed=value['coalesced_block_counts'];assert len(compressed)==value['decode_steps_after_first']+1 and len(compressed)>1
                assert compressed[0]==0 and all((v>0)==(policy=='candidate') for v in compressed[1:]) and len(set(compressed[1:]))==1
''')
needle="    metadata['wide_batch_prefill']=dict";assert source.count(needle)==1
source=source.replace(needle,"    metadata['scope']='Full native ABBA: original latency bundle, exact coalesced BF16 body blocks with64-thread decode; both VL B2 plans80. No fresh vLLM ratio.'\n    metadata['coalesced_body']=dict(threads=64,lossless_storage=True,fp32_arithmetic=True,original_raw_fallback=True,model_workloads_unchanged=True)\n"+needle)
destination=Path(__import__('sys').argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
