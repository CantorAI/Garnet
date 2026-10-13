# Rejected broad fused BF16 MLP decode

The tested GARNET_METAL_FUSED_MLP_DECODE=ON candidate combines gate/up BF16 projections and the
existing SiLU-times-up expression into one GPU kernel for one/two-row
decode. FP32 accumulation/lane/reduction order and original learned BF16
values are preserved. Chunk-origin projections and FP32-weight graphs use
their existing path. The candidate option defaults OFF; no prefill shortcut, reuse,
quantization or speculative prediction is introduced. The 128-thread
variant is used; the 64-thread component variant is not selected.

Two independent component runs pass all 96 bit-for-bit checks, including
first-layer real gate/up pairs, batch 1/2, irregular shapes and padded
strides. Early large component gains involve slower control samples and
are not treated as whole-model evidence. All eleven numerical/kernel,
serving, head and vision checks pass for the integrated candidate.

Actual operation diagnostics pass six complete requests and prove one
FusedMLP operation per decoder layer per generated step: 196 operations
for each text batch, 384 for coder and 648 for VL. Diagnostics serialize
operations and use historical chunk profiles, so their timings are not
serving speed measurements. Outputs remain identical to frozen controls.

Two sequential native ABBA runs pass all 144 timed complete requests across
all three models and batches 1/2. Each build/profile has two fresh workers,
three requests each, and four non-first warm samples. Dense uses the retained
finer planner (text 24/coder 32) on both sides; VL uses retained chunk 64/32
recurrence on both sides and fresh identical RGB/patch bytes. Caps, stop
rules, prompt/output IDs, weights and FP32 activations/KV are unchanged.

| Model | Batch | Initial decode ratio | Confirmation decode ratio | Confirmation TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 0.997 | 0.976 | 1.031 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.003 | 1.007 | 0.992 |
| Qwen3-0.6B | 2 | 0.996 | 0.998 | 0.973 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1.014 | 1.016 | 1.019 |
| Qwen3.5-0.8B VL | 1 | 1.014 | 1.011 | 1.003 |
| Qwen3.5-0.8B VL | 2 | 1.009 | 1.005 | 0.996 |

The candidate is rejected and its C++ integration/build flag are reverted.
Text batch 1 is 2.4% slower in the repeat and TTFT is 3.1% worse. Coder
batch-2 decode improves 1.4-1.6%, but its repeat TTFT is 1.9% worse. VL
batch-1 decode improves 1.1-1.4%; batch-2 gains only 0.5% in the repeat.
These small gains do not justify broad promotion with a text regression.
No material TTFT gain is established. First-use TTFT, initialization, peak
process RSS and full phase records are retained separately. Fresh workers
do not imply flushed filesystem/driver caches. GPU time overlaps wait time.

This is a native comparison. No new vLLM result or 2x/half-TTFT achievement
is established here. The full three-model goal remains active and unmet.
Only the executed models, prompts, caps and batch profiles are validated;
other Apple devices and production workloads remain unmeasured.

The serving kernel's source is exactly the bit-checked component kernel,
apart from its name. With the option OFF, the Metal shader is identical
to the previous shader except whitespace; both properties were checked
locally. Prior selected binaries and protected checkouts remain intact.

Candidate library SHA256: d14ee49e38414b2c49b73af173504323f13423ab4c6de8f5d5629d3faba6e6a5.

Control library SHA256: 56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4.

Archive SHA256: ad6b54c269a057b938329be26863b509d2aac57628992591b9cbeab86e695698.

Completed source, commands, terminal statuses, outputs, numerical references,
per-operation proof, process guards and raw timing/resource records are in
completed/. The component repeat waited for an unrelated xlang3 job to exit
naturally; loaded inference libraries were absent, but the guard remained
unchanged. SSH observation timeouts caused no job restart. Private tensor,
matrix and image bytes remain on the Mac with hashes in evidence-proof.json.

The current native source is restored byte-for-byte to the previous committed
source and the selected small-output library is unchanged. The archived
prototype source and separate candidate binary remain available for later
shape-specific investigation; no beneficial subset is claimed without execution.
Reproduce the integrated candidate from completed/source and
completed/build_fused_mlp_decode.py in a separate checkout/build. The
candidate build flag is present in that archived prototype.
