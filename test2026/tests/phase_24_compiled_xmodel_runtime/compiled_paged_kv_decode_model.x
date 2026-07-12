from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "qkv", "kind": "tensor"},
        {"name": "key_pages", "kind": "tensor"},
        {"name": "value_pages", "kind": "tensor"},
        {"name": "page_table", "kind": "tensor"},
        {"name": "context_length", "kind": "tensor"},
        {"name": "slot_position", "kind": "tensor"}
    ]
}


@T.fusion()
def Model(qkv, key_pages, value_pages, page_table, context_length, slot_position):
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * key_pages
    state = state * T.binary_op("paged_kv_bind_value_pages") * value_pages
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op("paged_kv_bind_context_length") * context_length
    state = state * T.binary_op("paged_kv_bind_slot_position") * slot_position
    return state * T.unary_op(
        "paged_kv_decode_bf16",
        page_size=4,
        q_heads=4,
        kv_heads=2,
        head_dim=8
    )
