"""Audit full native gates, actual layout counts and unchanged complete outputs."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_small_seed.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('small-seed-checks-20261015','interleaved-head-checks-20261015'),('small-seed-native-20261015','interleaved-head-native-20261015'),('build-small-screen-seed','build-interleaved-screen-index'),('build-serial-vision-prefetch','build-small-screen-seed'),('small-seed-original-source-20261015','interleaved-head-original-source-20261015'),('b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d','d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876'),("dependencies['pid']==99758","dependencies['pid']==33362")]:source=source.replace(old,new)
source=source.replace("assert 'METAL_SMALL_SCREEN_SEED' in meta['options']","assert 'METAL_INTERLEAVED_SCREEN_INDEX' in meta['options'] and 'METAL_SMALL_SCREEN_SEED' in meta['options']")
source=source.replace("assert row['small_screen_seed_enabled']==(policy=='candidate')","assert row['small_screen_seed_enabled']")
source=source.replace("row['screened_head_seed_rows']==[1024 if policy=='candidate' else 8192]*len(counts)","row['screened_head_seed_rows']==[1024]*len(counts)")
needle="            counts=row['screened_head_counts'];assert len(counts)==row['decode_steps_after_first']+1 and all(v==1 for v in counts)";assert source.count(needle)==1
source=source.replace(needle,needle+"\n            assert row['interleaved_screen_enabled']==(policy=='candidate')\n            assert row['interleaved_screen_counts']==[1 if policy=='candidate' else 0]*len(counts)")
source=source.replace("all(v['seed_rows']==1024 for v in values)","all(v['seed_rows']==1024 and v['interleaved_dispatches']==1 for v in values)")
source=source.replace("all(r['actual_seed_rows']==1024 for r in rows)","all(r['actual_seed_rows']==1024 and r['actual_interleaved'] for r in rows)")
source=source.replace('actual_candidate_seed_rows=1024','actual_interleaved=True,actual_candidate_seed_rows=1024')
exec(compile(source,__file__,'exec'))
