# Qwen3-1.7B

Pure-text Qwen3-1.7B TensorGraph programs. Runtime selection remains external
to the model; these files contain model semantics and paged-KV scheduling
boundaries only.

- `prefill.py`: prompt prefill and paged-KV population
- `decode.py`: latency-oriented single-request decode
- `decode_batch.py`: masked continuous-batch decode

The official two-shard BF16 checkpoint is accepted directly; no weight merge
or model conversion is required.

For native prompt-to-text generation, load `prefill.py` with
`frontend="qwen3_text"` and pass `prompt`, `enable_thinking`, and
`max_new_tokens` to `forward`.
