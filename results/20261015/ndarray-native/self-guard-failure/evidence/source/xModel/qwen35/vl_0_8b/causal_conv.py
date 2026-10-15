# SPDX-License-Identifier: Apache-2.0
"""One-token causal QKV convolution, with caller-owned per-request state."""
import garnet
from .tensor_compat import tensor
T = tensor()

@T.fusion(name="qwen35_causal_conv_step", role="linear_attention", boundary="required")
def Qwen35CausalConvStep(projected_qkv, weight, conv_state, active_rows):
    bound = projected_qkv * T.binary_op("qwen35_conv_bind_weight") * weight
    bound = bound * T.binary_op("qwen35_conv_bind_state") * conv_state
    bound = bound * T.binary_op("qwen35_conv_bind_active") * active_rows
    return bound * T.unary_op("qwen35_conv_step")

def output_from_packed(packed, kernel_size):
    return packed * T.unary_op("qwen35_conv_output", kernel_size=kernel_size)

def state_from_packed(packed, kernel_size):
    return packed * T.unary_op("qwen35_conv_state", kernel_size=kernel_size)
