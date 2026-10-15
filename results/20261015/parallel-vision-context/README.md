# Parallel FP32 vision value contraction

Two actual component trials pass the original numerical envelopes and show
lower time at the frozen 216-token, 12-head, 64-dimension shape. Independent
audit covers 42 archived files, 48 shape checks and 7680 timed submissions.
This does not establish whole-image TTFT, generated-output compatibility or
the full optimization goal.

The selected global-alpha library uses separate softmax and value operations;
its `GARNET_METAL_FUSED_VISION_CONTEXT` cache flag is OFF. The component control
is the earlier optional fused kernel, not that selected library. It retains
the serial 216-key FP32 contraction after the original SIMD softmax. The new
candidate splits the key range across four SIMD groups, stores FP32 partials
in 2048 bytes of threadgroup memory and combines them in FP32. Softmax,
normalization, input scores/values and output storage remain unchanged. This
changes summation order, so numerical compatibility does not mean bit identity.

| Batch, frozen shape | GPU fraction initial / repeat | Wall fraction initial / repeat |
| --- | ---: | ---: |
| 1 | 0.87985 / 0.83225 | 0.92502 / 0.87337 |
| 2 | 0.89172 / 0.90042 | 0.90058 / 0.91143 |

Fractions below one are faster. Each trial covers B1/B2, 4/24/216 tokens,
2/12 heads and dimensions 8/64, with four fresh input generations including
large score magnitudes. Every output and padding element is checked. Both
original and candidate satisfy the existing independent FP64 envelope
`5e-6 + 5e-5*abs(reference)`; candidate/control differences satisfy the same
existing envelope. No tolerance, arithmetic precision or model weight changes.
The frozen candidate's worst FP64 ratios are 0.0117424 and 0.0130210, below 1.
The changed results are not bit-identical. Component inputs are synthetic;
trained full-grid image embeddings are a separate required gate.

Each shape has common alternating warmup of at least 250 ms, two ABBA rounds,
three per-window warmups and 20 raw timed submissions per window. Shader and
pipeline compilation, allocation and validation are outside component timing.
Raw GPU/wall observations, original control source, generated candidate source,
compiler/macOS versions, process guards and recomputed means remain in evidence.
OS/driver caches were not flushed.

Owned component PIDs 60423 and 60805 terminated exit 0 after 13.2153 and
13.1141 seconds. They waited for the exact owned frontend confirmation and
never overlapped inference. Initial archive SHA-256:
`859ad5a3b03e5c730c12e8bfc80e51999e67fc72bcc1c667c6ea3c3b9f51224a`.
Combined archive SHA-256:
`0176cc262ae83d4288e2d15bdd3eaf8d6e623f63223bbd9279333f6c4b9d36ca`.

## Optional native integration under validation

`GARNET_METAL_PARALLEL_VISION_CONTEXT` defaults OFF and requires the existing
fused context and MPS score flags. It embeds the independently measured shader
body. Routing admits only 216 patch tokens, hidden width 768, 12 heads and
216/432 physical rows, with 32-lane SIMD, 128-thread and threadgroup-memory
checks. Other shapes or an unavailable pipeline retain the prior path. The
candidate build enables both fusion and parallel contraction; any measured
serving difference versus selected global-alpha therefore includes both changes.

A per-context counter resets at Begin/Abort; graph clear/run resets its
published counter, and only successful execution publishes it. The worker
records real per-image dispatch counts. Learned weights, image resolution,
FP32 activations/KV, prompt/caps/stops and decoder kernels remain unchanged.

Installation verified seven original source identities against `00f524e9`,
preserved their bytes and installed twelve pinned candidate files. Owned
validation PID 61469 at `logs/parallel-vision-checks-20261015` has completed
configure, build and CTest, and continues the original nineteen gates plus a
trained full-grid gate. The latter freshly encodes the canonical real image
twice per policy at B1/B2, checks every trained embedding against selected
global-alpha under the unchanged envelope, verifies twelve actual candidate
dispatches per image and zero published count after close. It is not yet a
passed gate or an inference speed claim.

Owned queue PID 64917 at `logs/parallel-vision-native-20261015` waits for exact
validation PID 61469, all twenty terminal successful stages and unchanged
installed source hashes. Only then does it run six sequential native ABBA
profiles. VL uses the same combined CPU frontend on both builds, with fresh
encoding and canonical patch/RGB hashes. Full generated token IDs, prompts,
stops, counts and flags must match before timings are accepted. Initialization,
first request after initialization, warm means and RSS remain separate.

The full three-model 2x decode / half-TTFT goal remains active and unmet. A
component speedup, successful build or queued serving study is insufficient
to change that status. The original selected libraries, GPT-OSS/Pi checkouts
and original Mac checkout are preserved.

```text
python3 queue_parallel_vision_context.py ROOT ROOT/logs/parallel-vision-context-20261015
python3 archive_parallel_vision_context.py ROOT ARCHIVE
python -X utf8 verify_parallel_vision_context.py RESULTS VERIFICATION_JSON
python -X utf8 make_parallel_vision_header.py
python -X utf8 prepare_parallel_vision_sources.py
python3 install_parallel_vision_sources.py ROOT
python3 launch_parallel_vision_checks.py ROOT ROOT/logs/parallel-vision-checks-20261015 checks
python3 queue_parallel_vision_profiles.py ROOT ROOT/logs/parallel-vision-native-20261015
```
