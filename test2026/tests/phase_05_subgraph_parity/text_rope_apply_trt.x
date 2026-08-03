from garnet import garnet

T = garnet.tensor()

qkv = tensor(shape=[4, 4096], dtype=tensor.float32)
cos = tensor(shape=[4, 128], dtype=tensor.float32)
sin = tensor(shape=[4, 128], dtype=tensor.float32)

output = qkv * T.binary_op("qwen3_vl_apply_text_rope", cos=cos, sin=sin)

print("Text RoPE apply TRT expression built.")
