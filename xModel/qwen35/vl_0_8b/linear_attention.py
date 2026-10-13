# SPDX-License-Identifier: Apache-2.0
"""Single-token Qwen3.5 GatedDeltaNet; returns output plus explicit states.

Packed row layout: [hidden output, flattened convolution history,
flattened recurrent state]. The serving caller extracts these tensors.
"""
import garnet
from .tensor_compat import tensor
T=tensor()

@T.fusion(name="qwen35_linear_attention_step",role="linear_attention",boundary="required")
def Qwen35LinearAttentionStep(hidden,conv_state,recurrent_state,active_rows,config,layer_idx,chunk=False):
    bound=hidden * T.binary_op("qwen35_conv_bind_state") * conv_state
    bound=bound * T.binary_op("qwen35_delta_bind_state") * recurrent_state
    bound=bound * T.binary_op("qwen35_conv_bind_active") * active_rows
    return bound * T.unary_op("qwen35_linear_chunk" if chunk else "qwen35_linear_step",
        weight_prefix="model.language_model.layers."+str(layer_idx)+".linear_attn",
        key_heads=config["linear_num_key_heads"],value_heads=config["linear_num_value_heads"],
        key_dim=config["linear_key_head_dim"],value_dim=config["linear_value_head_dim"],
        eps=config["rms_norm_eps"])

def unpack(packed,config):
    dimensions=dict(hidden_dim=config["hidden_size"],key_heads=config["linear_num_key_heads"],
        value_heads=config["linear_num_value_heads"],key_dim=config["linear_key_head_dim"],
        value_dim=config["linear_value_head_dim"],kernel_size=config["linear_conv_kernel_dim"])
    return [packed * T.unary_op("qwen35_linear_output",**dimensions),
            packed * T.unary_op("qwen35_linear_conv_state",**dimensions),
            packed * T.unary_op("qwen35_linear_recurrent_state",**dimensions)]
