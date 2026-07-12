from garnet import garnet

T = garnet.tensor()
T.set_backend("TensorRT")

GARNET_MODEL_SPEC = {
    "arguments": [
        {"name": "qkv", "kind": "tensor"},
        {"name": "key_pages", "kind": "tensor"},
        {"name": "value_pages", "kind": "tensor"},
        {"name": "page_table", "kind": "tensor"},
        {"name": "start_position", "kind": "tensor"}
    ]
}


@T.fusion()
def Model(qkv, key_pages, value_pages, page_table, start_position):
    state = qkv * T.binary_op("paged_kv_bind_key_pages") * key_pages
    state = state * T.binary_op("paged_kv_bind_value_pages") * value_pages
    state = state * T.binary_op("paged_kv_bind_page_table") * page_table
    state = state * T.binary_op("paged_kv_bind_slot_position") * start_position
    return state * T.unary_op(
        "paged_kv_prefill_write_bf16",
        page_size=4,
        q_heads=4,
        kv_heads=2,
        head_dim=8
    )
