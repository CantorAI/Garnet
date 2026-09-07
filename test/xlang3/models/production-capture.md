# Production Program Capture Coverage

Run with the built native module available:

```powershell
.\xlang3.exe <Garnet>/test/xlang3/models/run_production_capture.py
```

Verified on Windows Release with the actual `import garnet` package. No mock
modules, checkpoints, tokenizer files, or downloaded weights are used. The test
imports source files from the repository, not potentially stale deployed copies.

All 27 production `.py` counterparts imported. The following 19 entry points
each captured with one-layer and two-layer configurations:

| Model family | Entry points |
| --- | --- |
| Text 1.7B | Prefill, decode, batch decode |
| VL 2B | Full VL model, VL prefill, text prefill/decode/batch decode, vision pooler probe |
| ASR 0.6B | Prefill, decode |
| TTS 0.6B | Talker prefill/decode, code predictor, codec decode |
| TTS 1.7B | Talker prefill/decode, code predictor, codec decode |

The eight helper modules are imported and exercised through these roots. The VL
adapter's separate `Qwen3VisionPositionIds` and `Qwen3GetRopeIndex` functions also
capture independently, bringing the total to 38 root graphs and two helper
graphs. This is not an assertion that every optional branch of every helper was
executed.

The tests check graph dependency order, unique node IDs, required root fusion
boundaries, absence of a required boundary inside an atomic region, fresh root
invocation IDs, and increased operation counts for two layers. Shapes are tiny
symbolic profiles; weights are an empty dictionary because the production
programs name backend-loaded weights in operator attributes.

The first capture found real legacy `config.member` accesses incompatible with
the dictionary supplied by the native runtime. The `.py` programs now use Python
dictionary indexing. The 27 paired legacy `.x` files have been removed, and
model manifests now select `.py` entrypoints. Other manifest settings are preserved.

## Lowering Limits Found

Capture permits registered operator names without running a backend. It does
not validate pretrained numerics or prove that those operators can be lowered.
The captured inventory has 57 Garnet operators plus CPU addition.

A static string-dispatch audit found all 54 Garnet operators reached by the 19
root programs in TensorRT's implementation. Three operators emitted only by the
additional VL helpers were absent from both backends:

- `qwen3_vl_llm_vision_position_ids`
- `qwen3_vl_get_position_ids`
- `qwen3_vl_get_mrope_position_deltas`

OpenVINO additionally lacks dispatch strings for these 32 root operators:

```text
argmax_last_dim, concat_sequence, concat_tokens, last_token, layer_norm,
paged_kv_bind_active_mask, paged_kv_decode_masked_bf16, paged_kv_update_packed,
qwen3_asr_audio_attention_packed, qwen3_asr_audio_qkv_packed,
qwen3_asr_compact_audio_tokens, qwen3_asr_conv_subsample,
qwen3_asr_merge_audio_embeddings, qwen3_positions,
qwen3_tts_aligned_prompt_embedding, qwen3_tts_causal_conv1d,
qwen3_tts_causal_transconv1d, qwen3_tts_channel_scale,
qwen3_tts_decode_embedding, qwen3_tts_dense_attention_packed,
qwen3_tts_pack_hidden_logits, qwen3_tts_rvq_decode,
qwen3_tts_snake_beta, qwen3_tts_waveform,
qwen3_vl_apply_text_rope_packed, qwen3_vl_apply_vision_rope_packed,
qwen3_vl_deepstack_add, qwen3_vl_merge_visual_embeddings,
qwen3_vl_patch_embed_conv3d, qwen3_vl_patch_merger_shuffle,
qwen3_vl_pos_embed_interpolate, vision_varlen_attention_packed
```

String-dispatch presence is only a static coverage signal, not execution proof.
All 32 OpenVINO root gaps and the three helper gaps listed above were also
absent from the respective backend dispatch strings at pre-migration Git HEAD
`8698fe6413556a25656de94500e30a92e70fc882`. They are preexisting coverage limits,
not newly removed migration functionality. This migration preserves existing
backend feature coverage; implementing those previously missing operators is
outside this pass.
The separately verified tiny native models establish actual compilation and
execution for add, matmul, and registered ReLU on both backends. Full pretrained
Qwen inference remains unverified without model assets.
