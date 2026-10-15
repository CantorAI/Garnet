# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 recurrent attention component, shared by all backend profiles.

This is a component program, not a complete pretrained VLM entrypoint.
Inputs are FP32 Q/K/V after causal convolution and head expansion; decay
is already log-space and beta already sigmoid-transformed. Recurrent
state belongs to the caller. Batch rows have independent state.
"""
import garnet
from .tensor_compat import tensor
T = tensor()
GARNET_MODEL_SPEC = {"arguments": [
    {"name": "query", "kind": "tensor"},
    {"name": "key", "kind": "tensor"},
    {"name": "value", "kind": "tensor"},
    {"name": "log_decay", "kind": "tensor"},
    {"name": "beta", "kind": "tensor"},
    {"name": "recurrent_state", "kind": "tensor"},
    {"name": "active_rows", "kind": "tensor"}
]}

@T.fusion(name="qwen35_gated_delta_step", role="linear_attention", boundary="required")
def Qwen35GatedDeltaStep(query, key, value, log_decay, beta, recurrent_state, active_rows):
    state = query * T.binary_op("qwen35_delta_bind_key") * key
    state = state * T.binary_op("qwen35_delta_bind_value") * value
    state = state * T.binary_op("qwen35_delta_bind_decay") * log_decay
    state = state * T.binary_op("qwen35_delta_bind_beta") * beta
    state = state * T.binary_op("qwen35_delta_bind_state") * recurrent_state
    state = state * T.binary_op("qwen35_delta_bind_active") * active_rows
    return state * T.unary_op("qwen35_delta_step")

def output_from_packed(packed, key_dim, value_dim):
    return packed * T.unary_op("qwen35_delta_output", key_dim=key_dim, value_dim=value_dim)

def state_from_packed(packed, key_dim, value_dim):
    return packed * T.unary_op("qwen35_delta_state", key_dim=key_dim, value_dim=value_dim)
