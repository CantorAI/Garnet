"""Create a small original-layout checkpoint and independent dense CPU reference.

Run with CPython; uses only the standard library, no downloaded model assets.
"""
import argparse
import json
import math
import struct
import shutil
from pathlib import Path


def bf(x):
    value = struct.unpack('<I', struct.pack('<f', x))[0]
    value = (value + 0x7fff + ((value >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', value))[0]


def create(root):
    root.mkdir(parents=True, exist_ok=True)
    config = dict(num_hidden_layers=2, num_experts=5, experts_per_token=2,
                  vocab_size=64, hidden_size=32, intermediate_size=32,
                  swiglu_limit=7, head_dim=8, num_attention_heads=4,
                  num_key_value_heads=2, sliding_window=2, initial_context_length=4096,
                  rope_theta=150000, rope_scaling_factor=32, rope_ntk_alpha=1, rope_ntk_beta=32)
    entries, tensors, dense, data = {}, {}, {}, bytearray()

    def add(name, shape, values, dtype='BF16'):
        start = len(data)
        values = list(values)
        assert math.prod(shape) == len(values)
        if dtype == 'U8':
            data.extend(bytes(values))
        else:
            values = [bf(x) for x in values]
            for x in values:
                data.extend(struct.pack('<I', struct.unpack('<I', struct.pack('<f', x))[0])[2:])
        tensors[name] = values
        entries[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(data)])

    def floating(name, shape, scale=.075):
        add(name, shape, (math.sin(i * .47 + len(name)) * scale for i in range(math.prod(shape))))

    def packed(name, rows):
        blocks = [(i * 73 + len(name)) % 256 for i in range(rows * 16)]
        scales = [121 + i % 3 for i in range(rows)]
        experts, output = 5, rows // 5
        add(name + '.blocks', [experts, output, 1, 16], blocks, 'U8')
        add(name + '.scales', [experts, output, 1], scales, 'U8')
        lut = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
        dense[name] = [math.ldexp(lut[(blocks[r * 16 + c // 2] >> (4 * (c % 2))) & 15], scales[r] - 127)
                       for r in range(rows) for c in range(32)]

    floating('embedding.weight', [64, 32], .5)
    floating('unembedding.weight', [64, 32])
    add('norm.scale', [32], [1] * 32)
    for layer in range(2):
        p = f'block.{layer}'
        add(p + '.attn.norm.scale', [32], [1] * 32)
        add(p + '.mlp.norm.scale', [32], [1] * 32)
        floating(p + '.attn.qkv.weight', [64, 32])
        floating(p + '.attn.qkv.bias', [64], .03)
        floating(p + '.attn.out.weight', [32, 32])
        floating(p + '.attn.out.bias', [32], .03)
        floating(p + '.attn.sinks', [4], 1)
        floating(p + '.mlp.gate.weight', [5, 32])
        floating(p + '.mlp.gate.bias', [5], .1)
        floating(p + '.mlp.mlp1_bias', [5, 64], .02)
        floating(p + '.mlp.mlp2_bias', [5, 32], .02)
        packed(p + '.mlp.mlp1_weight', 5 * 64)
        packed(p + '.mlp.mlp2_weight', 5 * 32)
    header = json.dumps(entries, separators=(',', ':')).encode()
    header += b' ' * ((-len(header)) % 8)
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + data)
    (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')

    def norm(x, name):
        inverse = 1 / math.sqrt(sum(v * v for v in x) / len(x) + .00001)
        return [bf(v * inverse * s) for v, s in zip(x, tensors[name])]

    def linear(x, name, bias=None):
        w = tensors[name]
        result = [sum(v * w[row * len(x) + d] for d, v in enumerate(x))
                  for row in range(len(w) // len(x))]
        if bias:
            result = [bf(v + b) for v, b in zip(result, tensors[bias])]
        return [bf(v) for v in result]

    def rotary(qkv, position):
        out = qkv[:]
        low = 4 * math.log(4096 / (32 * 2 * math.pi)) / math.log(150000)
        high = 4 * math.log(4096 / (2 * math.pi)) / math.log(150000)
        for head in range(6):
            for d in range(8):
                ramp = min(1, max(0, (d % 4 - low) / (high - low)))
                inv = 150000 ** (-2 * (d % 4) / 8) * ((1 - ramp) + ramp / 32)
                cosine = bf(math.cos(position * inv) * (1 + .1 * math.log(32)))
                sine = bf(math.sin(position * inv) * (1 + .1 * math.log(32)))
                i = head * 8 + d
                rotated = -qkv[i + 4] if d < 4 else qkv[i - 4]
                out[i] = bf(bf(qkv[i] * cosine) + bf(rotated * sine))
        return out

    def moe(x, p):
        logits = linear(x, p + '.gate.weight', p + '.gate.bias')
        indices = sorted(range(5), key=lambda e: logits[e], reverse=True)[:2]
        denominator = sum(math.exp(logits[e] - logits[indices[0]]) for e in indices)
        probabilities = [bf(math.exp(logits[e] - logits[indices[0]]) / denominator) for e in indices]
        out = [0.] * 32
        for expert, probability in zip(indices, probabilities):
            up = dense[p + '.mlp1_weight'][expert * 64 * 32:(expert + 1) * 64 * 32]
            bias = tensors[p + '.mlp1_bias'][expert * 64:(expert + 1) * 64]
            projected = [bf(bf(sum(v * up[row * 32 + d] for d, v in enumerate(x))) + bias[row]) for row in range(64)]
            hidden = []
            for i in range(32):
                g, u = min(projected[2 * i], 7), min(7, max(-7, projected[2 * i + 1]))
                hidden.append(bf(bf(g * bf(1 / (1 + math.exp(-bf(1.702 * g))))) * bf(u + 1)))
            down = dense[p + '.mlp2_weight'][expert * 32 * 32:(expert + 1) * 32 * 32]
            bias = tensors[p + '.mlp2_bias'][expert * 32:(expert + 1) * 32]
            for row in range(32):
                out[row] += bf(bf(sum(v * down[row * 32 + i] for i, v in enumerate(hidden))) + bias[row]) * probability
        return [bf(v) for v in out]

    def reference(ids):
        xs = [tensors['embedding.weight'][i * 32:(i + 1) * 32] for i in ids]
        for layer in range(2):
            p = f'block.{layer}'
            qs = [rotary(linear(norm(x, p + '.attn.norm.scale'), p + '.attn.qkv.weight', p + '.attn.qkv.bias'), t)
                  for t, x in enumerate(xs)]
            outputs = []
            for t, x in enumerate(xs):
                attention = []
                for head in range(4):
                    first = max(0, t - 1) if layer % 2 == 0 else 0
                    scores = [sum(qs[t][head * 8 + d] * qs[s][32 + (head // 2) * 8 + d] for d in range(8)) / math.sqrt(8)
                              for s in range(first, t + 1)]
                    sink = tensors[p + '.attn.sinks'][head]
                    maximum = max(scores + [sink])
                    denominator = sum(math.exp(s - maximum) for s in scores + [sink])
                    for d in range(8):
                        attention.append(bf(sum(math.exp(score - maximum) * qs[s][48 + (head // 2) * 8 + d]
                                                for s, score in zip(range(first, t + 1), scores)) / denominator))
                projected = linear(attention, p + '.attn.out.weight', p + '.attn.out.bias')
                x = [bf(a + b) for a, b in zip(x, projected)]
                expert = moe(norm(x, p + '.mlp.norm.scale'), p + '.mlp')
                outputs.append([bf(a + b) for a, b in zip(x, expert)])
            xs = outputs
        return [linear(norm(x, 'norm.scale'), 'unembedding.weight') for x in xs]

    cases = [[1, 2, 3, 4], [6, 5, 4, 3]]
    sequence, generated = cases[0][:3], []
    for _ in range(3):
        logits = reference(sequence)[-1]
        token = max(range(len(logits)), key=lambda i: logits[i])
        generated.append(token)
        sequence.append(token)
    (root / 'expected.json').write_text(json.dumps({'ids': cases, 'logits': [reference(ids) for ids in cases],
                                                  'generated': generated}))
    (root / 'request.json').write_text(json.dumps({'input_ids': cases[0][:3], 'max_new_tokens': 3}))
    model_root = Path(__file__).resolve().parents[2] / 'xModel' / 'gpt_oss' / '120b'
    normal = {'id': 'gpt_oss', 'module': 'garnet_gpt_oss', 'abi': 1, 'backend': 'tensorrt',
              'operators': ['gpt_oss_round_bf16', 'gpt_oss_apply_yarn_rope_packed',
                            'gpt_oss_paged_attention', 'gpt_oss_moe_mxfp4',
                            'gpt_oss_rms_norm', 'gpt_oss_add_rms_norm']}
    invalid = {'abi_mismatch': [dict(normal, abi=2)],
               'backend_mismatch': [dict(normal, backend='openvino')],
               'missing_plugin': [dict(normal, id='missing')],
               'missing_operator': [dict(normal, operators=['gpt_oss_unknown'])],
               'invalid_list': {}, 'undeclared': []}
    for case, requirements in invalid.items():
        destination = root / 'cases' / case
        destination.mkdir(parents=True, exist_ok=True)
        for source in model_root.glob('*.py'):
            shutil.copyfile(source, destination / source.name)
        path = destination / 'prefill.py'
        lines = path.read_text().splitlines()
        lines = [('GARNET_MODEL_SPEC["requires"] = ' + repr({'operator_plugins': requirements}))
                 if line.startswith('GARNET_MODEL_SPEC["requires"]') else line for line in lines]
        path.write_text('\n'.join(lines) + '\n')
    print('Created synthetic original-layout checkpoint:', root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    create(parser.parse_args().destination)
