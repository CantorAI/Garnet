# FP32 vision softmax/value fusion: default-OFF experiment

The optional `GARNET_METAL_FUSED_VISION_CONTEXT` route retains the selected
batched MPS Q/K scores, then computes softmax and probability × value in one
threadgroup. It avoids a device-memory probability matrix and the following
compute dispatch. The score/softmax arithmetic and serial value accumulation
remain FP32. Checkpoint weights, prompts, image resolution and decoder policy
are unchanged.

This differs from the earlier rejected MPS value-contraction experiment:
it retains the serial contraction and removes the intermediate probability
output instead of packing values into another batched matrix multiplication.
At the serving shape, eliminating 12 graph-owned probability outputs removes
25.63 MiB / 51.26 MiB of logical FP32 elements at B1/B2. Those are calculated
buffer sizes, not measured RSS or unique GPU-memory reductions.

The flag defaults OFF and requires `GARNET_METAL_MPS_VISION_SCORES`. Graph
lowering selects it only for at most 1024 flattened rows, 512 patch tokens,
16 heads and dimension 128; other shapes use the existing two operations.
The serving library remains the preserved resident-KV control.

Two sequential component runs cover 24 shapes each: batch 1/2, patch tokens
4/24/216, heads 2/12 and dimensions 8/64. Four changed-input generations,
including large score magnitudes, are bit-identical to the extracted selected
softmax/value kernels. All independent FP64 checks pass; the largest error
uses 0.023403 of the established `5e-6 + 5e-5*abs(reference)` tolerance.
Output padding is preserved. Synthetic inputs are used in this component test.

| Serving-shaped component | Initial control / fused GPU ms | Initial ratio | Confirmation control / fused GPU ms | Confirmation ratio |
| --- | ---: | ---: | ---: | ---: |
| B1, 216 patches, 12 heads, dimension 64 | 1.2228 / 1.0264 | 1.1914× | 1.2238 / 1.0247 | 1.1944× |
| B2, 216 patches, 12 heads, dimension 64 | 2.2359 / 1.9719 | 1.1339× | 2.7587 / 2.4541 | 1.1241× |

The runs contain 384 ABBA timing records with 20 completed submissions per
record. These GPU command-buffer spans cover the softmax/value component,
excluding Q/K projection and scoring. They are not full-model speedups.
B2 absolute timing varies substantially between runs despite the similar
relative signal. Background guards and 0.5-second process monitors are retained.

The first two harness attempts failed before numerical checks: a missing
Objective-C++ closing brace, then an incomplete extracted struct declaration.
Both terminal failures and their source versions are preserved; corrected
component jobs complete with exit 0.

Commands and evidence are retained under `evidence/logs/`; generated execution
drivers preserve the exact standard checks and native ABBA workload. Source
snapshots and SHA-256 manifests preserve the control and candidate separately.
Initialization, first-use requests, warm requests and individual process RSS
are recorded separately. The first request in each worker is excluded from
warm means. The OS/driver caches are not flushed.

The full-frame workload uses the M4 / 16 GiB Mac mini, Qwen3.5-0.8B,
fresh 288×192 image preparation (216 patches and 54 visual tokens), the
unchanged frozen prompt and a 32-token cap. Batch 1 prefill uses `[80]`;
batch 2 uses `[32,32,16]`. Native workers run one at a time in ABBA order.
The control is the preserved selected library, not a new vLLM run.

All 18 configure/build/handoff stages pass, including native CTest, 48
exact-tail comparisons, all three private-KV suites, trained dense/recurrent
layers, output heads, synthetic vision and all 12 pretrained vision blocks.
Two native ABBA runs produce 48 complete frame outputs, all matching the
control's token IDs, stopping state, prompt IDs and exact image patch hash.

| Full-frame warm TTFT | Initial control → fused | Initial reduction | Confirmation control → fused | Confirmation reduction |
| --- | ---: | ---: | ---: | ---: |
| Batch 1 | 219.33 → 213.61 ms | 2.61% | 216.79 → 216.32 ms | 0.22% |
| Batch 2 | 388.28 → 383.47 ms | 1.24% | 389.20 → 383.62 ms | 1.43% |

B1's complete TTFT gain does not repeat. B2 shows a repeated modest gain,
well short of halving TTFT. Confirmation vision time is 59.44 → 58.04 ms
at B1 and 125.21 → 120.58 ms at B2. Other phases also vary; the entire TTFT
change cannot be assigned to the fused kernel alone. No further repeat is
justified solely to seek a more favorable B1 mean.

Confirmation decode throughput is 47.97 → 48.20 steps/s/row at B1 and
44.60 → 44.52 at B2. The decoder has no changed computation; these small
mixed differences do not establish a decode improvement. B2 aggregate
throughput is twice its per-row value for this equal-row workload.

Confirmation worker initialization is 2.966 → 2.953 seconds at B1 and
2.973 → 2.965 at B2. First-use frame TTFT, excluding initialization, is
340.68 → 342.66 ms and 566.42 → 564.19 ms respectively. Individual worker
peak RSS falls by 1.66 / 1.83 MiB in confirmation (7.31 / 7.44 MiB initially).
These small RSS shifts do not demonstrate the calculated logical-buffer
savings as measured unique GPU memory. RSS is from `/usr/bin/time -l`, not
an aggregate process-tree or shared-memory ownership measurement.

Decision: retain the optional route and validation helpers for further
investigation; keep it OFF and leave the primary serving library unchanged.
Its repeated component benefit and B2 frame benefit are bounded by this
fixed workload. This is not a broadly established model speedup.

The archive SHA-256 is
`a76aa5283d55e77287a6bcb636c78ae1bf0fc62fe3d0b0b06cd18fc8e7ad2be0`.
The candidate library SHA-256 is
`418985b013b9e121f23201391a754f3dfa5e548016b53237b098cca4edc91971`;
the preserved primary/control remains
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
`verify_vision_fused_context_evidence.py` verifies all 505 archived file hashes,
both retained harness failures, 48 shape/192 changed-input comparisons,
384 timing records, all terminal checks, actual compiler definitions,
both library identities and all 48 full-frame outputs/warm means.

The previous selected-frontend frame study measured B1 warm preprocessing
46.67 ms, vision 60.23 ms, prefill wall 108.85 ms and total TTFT 218.93 ms.
Its prefill profile reports roughly 100 ms of GPU time and 5.78 ms encoding
time for one warm request. Fusion therefore targets a limited fraction of
TTFT; prefill GPU work remains a larger optimization target. These numbers
come from the [prior frame study](../frame-frontend/README.md) and are not
measurements of this candidate.

The complete twofold decode / half-TTFT goal remains unmet. This experiment
does not establish a new vLLM-Metal comparison, stock BF16 wins, larger-model
behavior, long-context performance or general concurrent scheduling.
