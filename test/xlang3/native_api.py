# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import garnet as G
import tensor as T
import json


def fails(fn):
    try:
        fn()
    except Exception:
        return
    raise AssertionError("native call unexpectedly accepted invalid input")


def values(tensor, expected):
    actual = G.tensor_to_cpu(tensor).tolist()
    assert actual == expected, (actual, expected)


# Exercise the actual DLL package and its separately owned native instances.
first = G.QwenVLRequestContext()
second = G.QwenVLRequestContext()
first.prompt = "native smoke"
first.kv_handle = 42
assert first.prompt == "native smoke"
assert second.prompt == ""
assert first.stats()["kv_handle"] == 42
assert second.stats()["kv_handle"] == 0
model = G.model()
assert model.forward_request({})["status"] == "request_not_gpu_ready"
assert model.runtime_status()["ready"] is False
assert model.release_runtime() is True
assert json.loads(G.list_loaded_models_json())["models"] == []

cpu = G.tensor_from_host([1, 2, 3, 4, 5, 6], dtype="float32", shape=[2, 3], device="cpu")
alias = cpu
assert cpu.shape == (2, 3)
assert G.tensor_update_from_host(cpu, [6, 5, 4, 3, 2, 1]) is True
assert alias.tolist() == [6, 5, 4, 3, 2, 1]
fails(lambda: G.tensor_from_host([2147483648], dtype="int32", device="cpu"))
fails(lambda: G.tensor_from_host([1], shape=[4294967297], device="cpu"))
fails(lambda: G.tensor_from_host([1], device="not-a-device"))
empty = G.tensor_from_host([], dtype="int32", shape=[0, 3], device="cpu")
assert empty.shape == (0, 3)
assert G.tensor_update_from_host(empty, []) is True

strided = (cpu * T.unary_op("permute", axes=[1, 0])).eval()
assert strided.strides == (4, 12)
fails(lambda: G.tensor_update_from_host(strided, [0, 0, 0, 0, 0, 0]))
fails(lambda: G.tensor_add(strided, strided))
assert alias.tolist() == [6, 5, 4, 3, 2, 1]

bits = T.tensor([16256, 49152, 0], dtype=T.uint16)
bf_cpu = G.tensor_from_bfloat16_bits(bits, device="cpu")
values(bf_cpu, [1.0, -2.0, 0.0])

provider = G.tensor()
x = T.input("x", shape=[2, 3])
expression = x * provider.unary_op("relu")
nodes = T.graph(expression).inspect()
assert nodes[-1]["provider"] == "garnet"
assert nodes[-1]["name"] == "relu"
fails(lambda: G.tensor_add(x, x))
print("garnet-native-api-cpu-passed", flush=True)

# CUDA is required for this smoke test; missing CUDA must not silently pass.
gpu = G.tensor_to_gpu(cpu)
assert G.tensor_update_from_host(gpu, [1, 2, 3, 4, 5, 6]) is True
values(gpu, [1, 2, 3, 4, 5, 6])
values(cpu, [6, 5, 4, 3, 2, 1])
values(G.tensor_add(gpu, gpu), [2, 4, 6, 8, 10, 12])
values(G.tensor_last_row(gpu), [4, 5, 6])
values(G.tensor_to_bfloat16(gpu), [1, 2, 3, 4, 5, 6])
ids = G.tensor_from_host([1, 0], dtype="int64", device="cuda")
values(G.embedding(gpu, ids), [4, 5, 6, 1, 2, 3])
mask = G.tensor_from_host([0, 1], dtype="int64", device="cuda")
replacement = G.tensor_from_host([9, 8, 7], dtype="float32", shape=[1, 3], device="cuda")
values(G.replace_rows_by_mask(gpu, mask, replacement), [1, 2, 3, 9, 8, 7])
values(gpu, [1, 2, 3, 4, 5, 6])
zeros = G.tensor_from_host([0, 0], dtype="float32", device="cuda")
values(G.gelu_tanh(zeros), [0, 0])
print("garnet-native-api-cuda-passed", flush=True)
