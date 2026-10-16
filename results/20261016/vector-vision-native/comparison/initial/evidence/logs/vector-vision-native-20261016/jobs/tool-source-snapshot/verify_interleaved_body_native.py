"""Audit complete model output, actual decode body routes and original numerical gates."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_wide_batch_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-wide-batch-prefill','build-interleaved-bf16-body'),('build-interleaved-screen-index','build-wide-batch-prefill'),('wide-batch-prefill-checks-20261015','interleaved-body-native-checks-20261015'),('wide-batch-prefill-native-20261015','interleaved-body-native-20261015'),('wide-batch-prefill-original-source-20261015','interleaved-body-native-original-source-20261015'),("dependencies['pid']==69725","dependencies['pid']==49634")]:source=source.replace(old,new)
needle="assert proof['library_sha256']['build-wide-batch-prefill']=='837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471'";assert source.count(needle)==1
source=source.replace(needle,"assert proof['library_sha256']['build-wide-batch-prefill']=='9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b'")
source=source.replace("wide=model=='Qwen3.5-0.8B' and batch==2 and policy=='candidate'","wide=model=='Qwen3.5-0.8B' and batch==2")
source=source.replace("assert row['prefill_chunk_plan']==(([80] if wide else [32,32,16]) if batch==2 else [80])","assert row['prefill_chunk_plan']==[80]")
needle="            assert (row['prefill_profile'].get('wide_batch_prefill_dispatches',0)>0)==wide";assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert row['interleaved_body_enabled']==(policy=='candidate')
            body=row['interleaved_body_counts'];assert len(body)==row['decode_steps_after_first']+1 and len(body)>1 and body[0]==0
            assert all((v>0)==(policy=='candidate') for v in body[1:]) and len(set(body[1:]))==1
''')
needle="assert 'metal_wide_batch_prefill_reference' in ctext";assert source.count(needle)==1
source=source.replace(needle,needle+"\nassert 'metal_interleaved_bf16_body_reference' in ctext and 'METAL_INTERLEAVED_BF16_BODY' in json.loads((checks/'metadata.json').read_text())['options']")
exec(compile(source,__file__,'exec'))
