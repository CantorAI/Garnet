# Three-half activation prefill: rejected partial experiment

The isolated M4/16 GiB prototype splits each supported FP32 activation into
three FP16 components with per-row power-of-two scaling. It checks every
reconstructed input bit on the GPU, including signed zero. Each weight row
uses the existing exact BF16-to-scaled-FP16 converter. Three MPS products
produce FP32 intermediate outputs, which are combined and rescaled into FP32.
No approximate input is silently admitted. Wide-exponent inputs deliberately
fail reconstruction and require fallback; all three admission probes verify
this rejection. Metal helper shaders disable fast math for decomposition.

The control is one mixed FP32-input / exact scaled-FP16-weight MPS product
with FP32 output and rescaling, as used for compatible retained prefill
matrices. The candidate's three products reorder arithmetic; bit-exact
representations do not imply bit-exact products. The original output bound
5e-6 + 5e-5*abs(control) is unchanged. Inputs are deterministic synthetic
activations, and weights are actual first-layer checkpoint matrices.
This is not an independent full-model reference or matched vLLM comparison.

Text and coder complete 36 checks each at rows 24/32/64/80, including small
irregular matrices. VL completes eight checks before its QKV matrix reaches
a weight row outside the converter's supported range. The overall job exits
1, and is explicitly preserved as incomplete. No tolerance is widened and
no incompatible value is rounded to force acceptance. All 80 executed checks
pass, with worst tolerance ratio 0.296439417 and exact activation recovery.
The failed matrix is
model.language_model.layers.0.linear_attn.in_proj_qkv.weight.

| Tested MLP matrices | Rows | Control / candidate GPU ratio range |
| --- | ---: | ---: |
| Text gate/up/down | 24 | 0.341-0.392 |
| Text gate/up/down | 32 | 0.351-0.405 |
| Text gate/up/down | 64 | 0.357-0.424 |
| Text gate/up/down | 80 | 0.394-0.410 |
| Coder gate/up/down | 24 | 0.345-0.390 |
| Coder gate/up/down | 32 | 0.332-0.403 |
| Coder gate/up/down | 64 | 0.361-0.436 |
| Coder gate/up/down | 80 | 0.351-0.429 |

The candidate is about 2.3-3x slower on these MLP matrices. Each profile uses
two ABBA rounds, three warmups and twelve samples per position. GPU/wall
times include decomposition and recombination for the candidate. No other
inference runs simultaneously. There is no reason to integrate this slower
prototype, retry the unsupported VL matrix, or claim a speed improvement.

The serving source and selected library are unchanged. An encoder-reuse
change prepared locally during this investigation was discarded before
upload/build/run after the existing encoder-reuse report was found; that
prior experiment already showed negligible gain. Three local source files
were restored byte-exactly from their verified backups.

completed/ retains the terminal failure, partial records, all three admission
results, individual statuses, asset manifests and numerical/timing logs.
source/ retains the actual runner, shaders, native probe, converter and asset
collector. Raw tensors/binary remain on the Mac; hashes and sizes are in
evidence-proof.json. Transferred archive SHA256:
a5a8890cf6c328940ced2c9b736ebf51c9b7eca28a92c3d35bdf075c5a996586.

No new model performance or comparator result is established. The full
three-model 2x decode / half-TTFT objective remains active and unmet.
