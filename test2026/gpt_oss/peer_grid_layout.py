"""Exact grid protocol identity, admission storage, and unsafe-mode rejection."""
import copy
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/gpt_oss'))
from peer_group_layout import peer_group_layout,validate_peer_group_layout
from resident_budget import normalize_kernel_environment

for ctas in (64,128,188):
    env={'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
         'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':str(ctas)}
    with patch.dict(os.environ,env,clear=True):old=peer_group_layout(512,8,2880)
    with patch.dict(os.environ,dict(env,GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS='0'),clear=True):
        assert peer_group_layout(512,8,2880)==old
    with patch.dict(os.environ,dict(env,GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS='1'),clear=True):
        new=peer_group_layout(512,8,2880)
    assert new['schema']==3 and new['protocol']=='owned-mapped-bf16-peer-grid-v3' and new['grid_signals']==1
    assert new['phase_elements']==old['phase_elements'] and new['capacity_elements']==old['capacity_elements']
    assert validate_peer_group_layout(new,512,8,2880)==validate_peer_group_layout(old,512,8,2880)
    for key,value in [('schema',2),('protocol',old['protocol']),('grid_signals',True),
                      ('grid_signals',0),('grid_signals','1'),('mapped_host_bytes',0),('ctas',32)]:
        bad=copy.deepcopy(new);bad[key]=value
        try:validate_peer_group_layout(bad,512,8,2880)
        except ValueError:pass
        else:raise AssertionError((key,value))
for group,grid in [('0','1'),('1','2'),('1','true'),('1',''),('invalid','1')]:
    with patch.dict(os.environ,{'GARNET_GPT_OSS_BF16_PEER_GROUP':group,
        'GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS':grid},clear=True):
        try:peer_group_layout(512,8,2880)
        except ValueError:pass
        else:raise AssertionError((group,grid))
assert normalize_kernel_environment({})['GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS']=='0'
assert normalize_kernel_environment({'GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS':'1'})!=normalize_kernel_environment({})
assert normalize_kernel_environment({})['GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS']=='256'
for threads in ('256','512'):
    with patch.dict(os.environ,{'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
        'GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS':threads},clear=True):
        assert peer_group_layout(512,8,2880) is not None
for group,threads in [('1','128'),('1','512x'),('0','512')]:
    with patch.dict(os.environ,{'GARNET_GPT_OSS_BF16_PEER_GROUP':group,
        'GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS':threads},clear=True):
        try:peer_group_layout(512,8,2880)
        except ValueError:pass
        else:raise AssertionError((group,threads))
for ctas in (64,128,188):
    for threads in (256,512):
        for grid in (0,1):
            with patch.dict(os.environ,{
                'GARNET_GPT_OSS_BF16_PEER_GROUP':'1',
                'GARNET_GPT_OSS_BF16_PEER_GROUP_CTAS':str(ctas),
                'GARNET_GPT_OSS_BF16_PEER_GROUP_THREADS':str(threads),
                'GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS':str(grid)},clear=True):
                layout=peer_group_layout(512,8,2880)
            if threads==512:
                assert layout['schema']==4 and layout['threads_per_cta']==512
                assert layout['protocol']==('owned-mapped-bf16-peer-grid-threads-v1'
                    if grid else 'owned-mapped-bf16-peer-threads-v1')
            elif grid:
                assert layout['schema']==3 and layout['protocol']=='owned-mapped-bf16-peer-grid-v3'
            else:
                assert layout['schema']==2 and layout['protocol']=='owned-mapped-bf16-peer-v2'
            assert validate_peer_group_layout(layout,512,8,2880)==layout['owned_bytes_per_rank']+layout['mapped_host_bytes']
            assert layout['owned_bytes_per_rank']==512*8*2880*2+ctas*132
            assert layout['mapped_host_bytes']==ctas*512
            wrong_schema=2 if layout['schema'] in (3,4) else 3
            for key,value in [('schema',wrong_schema),('protocol','invalid-peer-protocol'),
                              ('threads_per_cta',256),('threads_per_cta',1024)]:
                bad=copy.deepcopy(layout);bad[key]=value
                try:validate_peer_group_layout(bad,512,8,2880)
                except ValueError:pass
                else:raise AssertionError(('thread descriptor mutation',key,value,layout))
print('GRID_PROTOCOL_STORAGE_AND_REJECTION_PASS CTA64/128/188 DEFAULT_UNCHANGED')
