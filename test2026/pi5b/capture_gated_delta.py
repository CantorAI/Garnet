"""Capture the actual Qwen3.5 component with native Garnet/XLang3."""
import importlib
import os
import sys
import garnet
import tensor
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
module = importlib.import_module("xModel.qwen35.vl_0_8b.gated_delta")
@module.T.fusion(name="delta_capture_probe", boundary="required")
def probe(query, key, value, decay, beta, state, active):
    packed = module.Qwen35GatedDeltaStep(query, key, value, decay, beta, state, active)
    return [module.output_from_packed(packed, 7, 5),
            module.state_from_packed(packed, 7, 5)]

for batch in (1, 2):
    shapes = [[batch, 2, 7], [batch, 2, 7], [batch, 2, 5], [batch, 2],
              [batch, 2], [batch, 2, 7, 5], [batch]]
    args = [tensor.input("delta_" + str(batch) + "_" + str(i), shape=shape,
                         dtype=tensor.float32) for i, shape in enumerate(shapes)]
    nodes = probe(*args).inspect()
    seen = set()
    for node in nodes:
        for dependency in node["inputs"]:
            assert dependency in seen, "non-topological delta graph"
        assert node["id"] not in seen
        seen.add(node["id"])
    operations = [n for n in nodes if n["provider"] == "garnet"]
    assert len(operations) == 9, str(operations)
    print("qwen35 gated delta capture batch", batch, "operators",
          len(operations), "PASS", flush=True)
