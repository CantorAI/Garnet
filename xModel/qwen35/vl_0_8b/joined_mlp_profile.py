"""Admission for the experimental joined NVFP4 full decoder; no GPU imports."""
def validate_joined_mlp_profile(enabled,batch,capacity,chunk_size,external_compressed_weights,embedded_projection_weights=False,join_input_projections=False):
    if type(enabled) is not bool:
        raise ValueError('Joined MLP opt-in must be a boolean')
    if not enabled:
        return
    if type(batch) is not int or batch!=1 or type(capacity) is not int or not 32<=capacity<=4096 or type(chunk_size) is not int or chunk_size not in (4,128):
        raise ValueError('Joined MLP study requires B1/cap32..4096 and T4/T128')
    if external_compressed_weights is not True or embedded_projection_weights or join_input_projections:
        raise ValueError('Joined MLP requires only the shared compressed pool')
