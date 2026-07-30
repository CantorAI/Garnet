from garnet import garnet

T = garnet.tensor()

qkv = tensor(shape=[4, 3072], dtype=tensor.float32)
weights = {}
T.set_weights(weights)

output = qkv * T.unary_op("vision_attention_core")

print("Vision attention core TRT expression built.")
