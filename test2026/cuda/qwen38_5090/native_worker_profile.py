"""Validate research worker profiles before loading GPU resources."""

def resolve_profile(env):
    def flag(name):
        value=env.get(name,'0')
        if value not in ('0','1'):raise ValueError('Invalid '+name)
        return value=='1'
    packed=flag('GARNET_STUDY_PACKED')
    shared=flag('GARNET_STUDY_SHARED_COMPRESSED')
    resident=flag('GARNET_STUDY_RESIDENT_VISION')
    reuse_head=flag('GARNET_STUDY_REUSE_VERIFIED_HEAD')
    joined_mlp=flag('GARNET_STUDY_JOINED_MLP')
    depth=int(env.get('GARNET_STUDY_MTP_DEPTH','0'))
    chunk=int(env.get('GARNET_STUDY_MTP_CHUNK','4' if shared else '16'))
    target=4 if shared else 16
    if shared and not packed:raise ValueError('Shared compressed weights require packed mode')
    if depth not in (0,1,3,7) or depth and not packed:raise ValueError('Invalid MTP depth/profile')
    if chunk not in (4,8,16) or depth and not depth+1<=chunk<=target:raise ValueError('Invalid MTP draft chunk')
    if reuse_head and not (shared and depth==3):raise ValueError('Verified head reuse study requires shared MTP3')
    if joined_mlp and not (shared and depth==3 and not reuse_head):raise ValueError('Joined MLP HTTP study requires shared MTP3 with head reuse OFF')
    return dict(packed=packed,shared=shared,resident=resident,reuse_head=reuse_head,joined_mlp=joined_mlp,depth=depth,draft_chunk=chunk,
                decode_chunk=target,prefill_chunk=128 if shared else 16)
