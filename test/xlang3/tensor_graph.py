import tensor as T
import garnet_capture_test as G

P = G.tensor()


@P.fusion(name="decoder_layer", role="layer", atomic=True)
def layer(x, weight):
    norm = x * P.unary_op("rms_norm", weight=weight, eps=0.000001)
    return norm * P.binary_op("linear", weight_name="q_proj") * weight


@P.fusion(name="qwen_capture", boundary="required")
def model(x, weight):
    for i in range(3):
        x = layer(x, weight)
    return {"last": x, "alias": [x]}


x = T.input("x", shape=[2, 4])
weight = T.input("weight", shape=[4, 4], dtype=T.bfloat16)
graph = model(x, weight)
nodes = G.capture(graph)
ops = [node for node in nodes if node["provider"] == "garnet"]
assert len(ops) == 6
layer_ids = []
for i in range(3):
    norm = ops[i * 2]
    linear = ops[i * 2 + 1]
    assert norm["name"] == "rms_norm"
    assert norm["operand_count"] == 1
    assert len(norm["inputs"]) == 2
    assert norm["attributes"]["weight"] is weight
    assert norm["inputs"][1] is weight
    assert linear["operand_count"] == 2
    assert linear["attributes"]["weight_name"] == "q_proj"
    assert norm["ordered"] and linear["ordered"]
    assert len(norm["regions"]) == 2
    assert norm["regions"][1]["atomic"] is True
    assert norm["regions"][1]["function"] == "layer"
    assert norm["regions"][1]["id"] == linear["regions"][1]["id"]
    layer_ids.append(norm["regions"][1]["id"])
assert len(set(layer_ids)) == 3
partitions = G.partitions(graph)
assert [op["partition"] for op in partitions["operations"]] == [0, 0, 1, 1, 2, 2]
assert len(partitions["regions"]) == 4
assert partitions["regions"][0]["inclusive_operations"] == 6
assert partitions["regions"][0]["input_count"] == 2
assert partitions["regions"][0]["output_count"] == 1
assert [region["operations"] for region in partitions["regions"]] == [0, 2, 2, 2]
outputs = G.outputs(graph)
assert outputs["last"] is outputs["alias"][0]
last_id = outputs["last"].id
outputs["last"] = x
assert G.outputs(graph)["last"].id == last_id
assert G.capture(graph)[-1]["id"] == nodes[-1]["id"]
assert len([node for node in nodes if node["name"] == "input"]) == 2
assert set(node["attributes"]["name"] for node in nodes if node["name"] == "input") == {"x", "weight"}
next_graph = model(x, weight)
print("garnet-repeated-fusion-captured", flush=True)
next_ops = [node for node in G.capture(next_graph) if node["provider"] == "garnet"]
assert next_ops[0]["regions"][0]["id"] != ops[0]["regions"][0]["id"]
assert next_ops[0]["regions"][1]["id"] not in layer_ids

@P.fusion(atomic=True)
def unnamed_layer(x):
    return x * P.unary_op("identity")

unnamed = G.capture(unnamed_layer(x))[-1]
assert unnamed["regions"][0]["name"] == "unnamed_layer"
assert unnamed["regions"][0]["function"] == "unnamed_layer"

# Factories/expressions own their registrations independently of the provider.
provider = G.tensor()
print("garnet-second-provider-created", flush=True)
operator = provider.unary_op("retained")
del provider
print("garnet-provider-released", flush=True)
assert G.capture(T.graph(x * operator))[-1]["name"] == "retained"

# Composition and attributes are retained, not evaluated or stringified.
extra = x + x
attrs = {"items": [extra]}
expression = x * P.unary_op("bind", extra=attrs)
attrs["items"][0] = weight
captured = G.capture(T.graph(expression))
assert captured[-1]["attributes"]["extra"]["items"][0] is extra
assert captured[-1]["inputs"][1] is extra
assert captured[-2]["provider"] == "cpu"

failed = False
try:
    graph.run({"x": T.tensor(shape=[2, 4]), "weight": T.tensor(shape=[4, 4], dtype=T.bfloat16)})
except Exception:
    failed = True
assert failed  # Garnet operators require backend lowering, not a fake CPU result.

# CPU remains an independent backend for ordinary arithmetic.
cpu = T.graph((x + x) * T.unary_op("relu"))
assert cpu.run({"x": T.tensor([[-1, 2, 3, 4], [5, 6, 7, 8]])}).tolist() == [0, 4, 6, 8, 10, 12, 14, 16]
print("garnet-python-expression-capture-passed")
