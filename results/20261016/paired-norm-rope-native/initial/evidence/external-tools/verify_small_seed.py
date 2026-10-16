"""Audit all original gates and actual 1024-row native seed use and recovery."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_serial_vision.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('serial-vision-checks-20261015','small-seed-checks-20261015'),('serial-vision-native-20261015','small-seed-native-20261015'),('build-serial-vision-prefetch','build-small-screen-seed'),('build-screened-prefill-reset','build-serial-vision-prefetch'),('serial-vision-original-source-20261015','small-seed-original-source-20261015'),('75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e','b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d'),("dependencies['pid']==64980","dependencies['pid']==99758")]:source=source.replace(old,new)
source=source.replace("assert 'METAL_SERIAL_VISION_PREFETCH' in meta['options']","assert 'METAL_SMALL_SCREEN_SEED' in meta['options'] and 'METAL_SERIAL_VISION_PREFETCH' in meta['options']")
source=source.replace("assert row['serial_vision_prefetch_enabled']==(policy=='candidate')","assert row['serial_vision_prefetch_enabled']")
source=source.replace("row['vision_prefetch_dispatches']==(vision_config['depth'] if policy=='candidate' else 0)","row['vision_prefetch_dispatches']==vision_config['depth']")
needle="            counts=row['screened_head_counts'];assert len(counts)==row['decode_steps_after_first']+1 and all(v==1 for v in counts)";assert source.count(needle)==1
source=source.replace(needle,needle+"\n            assert row['small_screen_seed_enabled']==(policy=='candidate')\n            assert row['screened_head_seed_rows']==[1024 if policy=='candidate' else 8192]*len(counts)")
source=source.replace("    values=p['checks'];assert len(values)==17","    values=p['checks'];assert len(values)==17 and all(v['seed_rows']==1024 for v in values)")
needle="    if screened:recovered+=2";assert source.count(needle)==1
source=source.replace(needle,"    if screened:assert all(r['actual_seed_rows']==1024 for r in rows)\n"+needle)
source=source.replace('trained_full_grid_profiles=4','actual_candidate_seed_rows=1024,trained_full_grid_profiles=4')
exec(compile(source,__file__,'exec'))
