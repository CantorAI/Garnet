"""Audit native serial-vision full grids, original gates, counters and raw ABBA."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_screened_prefill_reset.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('screened-prefill-reset-checks-20261015','serial-vision-checks-20261015'),('screened-prefill-reset-native-20261015','serial-vision-native-20261015'),('build-screened-prefill-reset','build-serial-vision-prefetch'),('build-screened-head','build-screened-prefill-reset'),('screened-head-original-source-20261015','serial-vision-original-source-20261015'),('2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23','75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e'),('len(stages)==26','len(stages)==27'),("dependencies['pid']==29706","dependencies['pid']==64980"),("dependencies['completed_stages']==26","dependencies['completed_stages']==27")]:source=source.replace(old,new)
source=source.replace("names={'screened-real-heads'","names={'vision-full-grid','screened-real-heads'")
source=source.replace("assert 'METAL_SCREENED_HEAD' in meta['options']","assert 'METAL_SERIAL_VISION_PREFETCH' in meta['options'] and 'METAL_SCREENED_HEAD' in meta['options']")
source=source.replace("row['prefill_stream_enabled']==(policy=='candidate') and row['dirty_prefix_reset_enabled']==(policy=='candidate')","row['prefill_stream_enabled'] and row['dirty_prefix_reset_enabled']")
source=source.replace(" if policy=='candidate' else 0",'')
source=source.replace("if policy=='candidate':\n                kinds=","if row['dirty_prefix_reset_enabled']:\n                kinds=")
needle="                assert row['frontend_policy']=='combined' and row['frontend_route']=='threaded-striped'";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                assert row['serial_vision_prefetch_enabled']==(policy=='candidate')\n                vision_config=json.loads((root/'models'/model/'config.json').read_text())['vision_config']\n                assert row['vision_prefetch_dispatches']==(vision_config['depth'] if policy=='candidate' else 0)")
needle="trial=root/'logs'/native_trial";assert source.count(needle)==1
source=source.replace(needle,'''import numpy as np
grid=checks/'full-grid';full=json.loads((grid/'summary.json').read_text())
assert len(full['profiles'])==4 and {(p['batch'],p['profile']) for p in full['profiles']}=={(b,s) for b in (1,2) for s in ('small','full')}
for p in full['profiles']:
    b=p['batch'];label=p['profile'];arrays=[]
    assert p['bitwise_equal'] and p['fresh_encodes_per_policy']==2
    assert p['expected_candidate_dispatches']==(12 if label=='full' else 0)
    for policy in ('control','candidate'):
        stem=policy+'-'+label+'-b'+str(b);raw=json.loads((grid/(stem+'.json')).read_text())
        fixture=json.loads((grid/(stem+'-fixture.json')).read_text())
        assert raw['enabled']==(policy=='candidate') and raw['counts']==[p['expected_candidate_dispatches'] if policy=='candidate' else 0]*2
        assert fixture['batch']==b and fixture['expected_prefetch_dispatches']==p['expected_candidate_dispatches']
        a=np.asarray(raw['values'],dtype='<f4');assert a.size==p['values'] and np.all(np.isfinite(a))
        assert hashlib.sha256(a.tobytes()).hexdigest()==p['embedding_sha256'];arrays.append(a)
        from pathlib import PurePosixPath
        patch_name=str(PurePosixPath(fixture['patch_file']).relative_to('/Users/zsdx/garnet-metal-investigation'))
        assert proof['omitted'][patch_name]['sha256']==p['patch_sha256']
    assert np.array_equal(arrays[0].view('<u4'),arrays[1].view('<u4'))
'''+needle)
source=source.replace('full_goal_met=False','trained_full_grid_profiles=4,full_goal_met=False')
exec(compile(source,__file__,'exec'))
