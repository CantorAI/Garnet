# Qwen VL Precision Diagnostic

`qwen_vl_logits_parity.py` reproduces the WorldFusion room-inventory case through
the current native API. It intentionally reuses that application's loader and
prompt so the production input profile is unchanged. It does not alter runtime
code or substitute reference inference for native execution.

Use CPython 3.14 with the XLang3 bridge for `native`, and the isolated Torch
environment for `reference`. Run them sequentially to avoid competing for GPU
memory. Supply shared-workspace output paths for all artifacts and caches:

```text
python qwen_vl_logits_parity.py native --weights <weights> --image <prepared-image> --cache <engine-cache> --reference <reference-json> --pixels <native-pixels.npy> --output <native.npy> --step 31
python qwen_vl_logits_parity.py reference --weights <weights> --image <prepared-image> --cache <engine-cache> --reference <reference-json> --pixels <native-pixels.npy> --output <reference.npy> --step 31 --compare-native <native.npy>
```

The reference JSON contains `input_ids`, `image_grid_thw`, and `token_ids`.
The native side asserts exact prompt and image-pixel equality. Both backends are
fed the same generated-token prefix before exporting the next-token logits.
`--compare-native` saves finite-value checks and numerical metrics alongside
the reference array. These are diagnostic metrics, not a blanket parity pass.

Add `--fp32-head` to match Garnet's intentional FP32 language-model head while
leaving reference embeddings and transformer layers in BF16. Add
`--generate-tokens 1024` instead of `--compare-native` to record free-running
greedy output as JSON. The reference head must be detached from tied embeddings
before changing its dtype; otherwise the experiment changes input precision too.

## Observed Windows Result

For the prepared IMG_8301 room image, the first native/BF16-reference divergence
is generated token index 31 (zero-based). The first 31 tokens agree.

| Candidate | Native FP32 Head | Reference BF16 Head | Reference FP32 Head |
| --- | ---: | ---: | ---: |
| token 22 (`7`) | 27.501762 | 27.5 | 27.480034 |
| token 23 (`8`) | 27.511282 | 27.5 | 27.513079 |

The BF16 reference ties and selects the lower token ID. Both FP32 heads select
token 23. The final native versus BF16-reference probe has RMSE 0.03211 and
cosine similarity 0.9999439. FP32-head reference RMSE is 0.03174. An earlier
native probe had RMSE 0.03615; neither bitwise reproducibility nor exhaustive
numerical parity is asserted by this diagnostic.

Unmodified BF16 reference generation terminates. Reference generation with the
FP32 head also repeats object records and exhausts 1024 tokens without EOS,
as native greedy generation does (the repeated records differ). Therefore this
observed repetition is not unique to Garnet's sampler, bridge, or KV cache.
This does not rule out unrelated numerical defects elsewhere in the model.

No production precision change is justified by this case. Keep Garnet's existing
FP32 head and configurable generation controls; do not round logits merely to
make one fixture match. Exhaustive layer parity and broader dataset quality are
separate validation work.
