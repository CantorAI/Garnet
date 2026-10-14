"""Full-output audit plus explicit rotating-mask, ordinals and sample weights."""
from pathlib import Path
source=Path(__file__).with_name('verify_counter_trace_profiles.py').read_text()
source=source.replace('counter-trace-file-profiles-20261015','counter-trace-sparse-profiles-20261015').replace('counter-trace-file-source','counter-trace-sparse-source')
needle="            assert [e['index'] for e in by.get((c['context'],c['command']),[])]==list(range(c['samples']))"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert c['sampling_stride']==31 and c['encoded_total']>=c['samples']
            captured=by.get((c['context'],c['command']),[])
            ordinals=[e['encoder_ordinal'] for e in captured]
            assert ordinals==sorted(set(ordinals)) and all(0<=n<c['encoded_total'] for n in ordinals)
            for e in captured:
                head=e['op'].startswith('bf16:') and int(e['op'].split(':')[-1])>100000
                assert e['sampling_weight']==(1 if head else 31)
                assert head or (e['encoder_ordinal']+c['command'])%31==0
''')
source=source.replace('all_counter_records_contiguous=True,','all_counter_records_contiguous=True,sparse_mask_ordinals_and_weights_verified=True,')
exec(compile(source,__file__,'exec'))
