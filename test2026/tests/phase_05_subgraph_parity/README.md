# Phase 05: Qwen-VL Subgraph Parity

This phase compares isolated Garnet subgraphs against small reference tensor dumps before full Qwen-VL execution.

Initial priority:

1. `Qwen3TextMLP`
2. `VisionMLP`
3. `rms_norm`
4. `layer_norm`
5. multimodal merge
6. attention projection + RoPE
7. one decoder layer

## Text MLP Reference

`test_text_mlp_reference.py` creates a deterministic tiny Qwen3TextMLP reference dump:

```text
gate = x @ W_gate.T
up = x @ W_up.T
hidden = silu(gate) * up
output = hidden @ W_down.T
```

Artifacts are written under:

```text
test2026/artifacts/qwen_vl_subgraphs/
```

Generated artifacts are ignored by git.

## Garnet Parity

`test_text_mlp_garnet.py` runs a tiny TextMLP through Garnet/TensorRT when `RUN_GARNET_TEXT_MLP_PARITY=1`.

`test_vision_mlp_reference.py` and `test_vision_mlp_garnet.py` cover the next vision block:

```text
output = gelu_tanh(x @ W_fc1.T + b_fc1) @ W_fc2.T + b_fc2
```
