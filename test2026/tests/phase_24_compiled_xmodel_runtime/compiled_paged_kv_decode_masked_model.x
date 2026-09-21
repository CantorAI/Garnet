# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

from garnet import garnet

T = garnet.tensor()

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "qkv", "kind": "tensor"},
        {"name": "key_pages", "kind": "tensor"},
        {"name": "value_pages", "kind": "tensor"},
        {"name": "page_table", "kind": "tensor"},
        {"name": "context_length", "kind": "tensor"},
        {"name": "slot_position", "kind": "tensor"},
        {"name": "active_mask", "kind": "tensor"}
    ]
}


@T.fusion()
def Model(qkv, key_pages, value_pages, page_table, context_length,
          slot_position, active_mask):
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * key_pages
    state = state * T.binary_op("paged_kv_bind_value_pages") * value_pages
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op(
        "paged_kv_bind_context_length") * context_length
    state = state * T.binary_op(
        "paged_kv_bind_slot_position") * slot_position
    state = state * T.binary_op("paged_kv_bind_active_mask") * active_mask
    return state * T.unary_op(
        "paged_kv_decode_masked_bf16",
        page_size=16,
        q_heads=16,
        kv_heads=8,
        head_dim=128
    )
