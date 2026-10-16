# K32 N64 block packed prefill rejected by numerical gate

Actual Mac controller PID28401 exited0 after72.469517seconds. Both private
FP32 SIMD tile policies fail the original comparison envelope:6 of48 profiles
across75,601,920 FP32 output values. No timing, confirmation or native integration
was admitted. The matched three-model goal remains active and unchanged.

## Hypothesis and control

The earlier four-output interleaved layout makes neighboring output groups
stride across K. This experiment adds a full BF16 copy in N64/K32 blocks,
K-major inside each block, so a threadgroup loads its contiguous2,048 weights.
Policies1/2 use M32/M64, N64,K32,256threads,32-wide SIMD, and FP32 matrix
accumulation. Decode remains original. BF16 words expand exactly by shifting16;
FP32 inputs, checkpoint values, output/input limits and original key-projection
policy remain unchanged. The result scratch reuses input shared storage after
the final K barrier; static scratch sizes are16/24KiB. M64 skips inactive eight-row
accumulator bands with group-uniform conditions.

The control is the accepted d52 context with its original selected MPS, BF16
and key routes. Both sides use the extra weight allocation; setup is outside
any potential GPU timing. This is a private context, not a selected Garnet
library. Pipeline compilation uses the exact accepted defines with fast math
policy inherited unchanged. Original BF16 words including padded zeros are
roundtripped on CPU for every allocated matrix before upload.

## Complete numerical gate

All29 checkpoint-verified matrices run:7text,7coder,15VL (linear layer0 and
full-attention layer3);20 use the new layout. Physical rows are24/44,26/52,
80/160. Each batch/policy runs four original deterministic activation patterns
against FP64 and control. Preserve gamma based on ceil(K/32)+32, FP32 epsilon,
absolute floor1e-7, and control comparison5e-6+5e-5*abs(control). No bound changed.

The worst FP64 ratio is0.075892469 for both original control and candidate,
but the control comparison reaches1.905077547 against a maximum of1. These are
separate criteria: a forward-error pass does not excuse comparison failures.
Each row below contains four patterns per policy; zero comparison means all
recorded outputs match the control numerically for those patterns.

| Model | Batch | M32 failed | M32 worst comparison ratio | M64 failed | M64 worst comparison ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 1 | 1.413227 | 1 | 1.413227 |
| Qwen3-0.6B | 2 | 0 | 0.000000 | 0 | 0.000000 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1 | 1.905078 | 1 | 1.905078 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1 | 1.728992 | 1 | 1.728992 |
| Qwen3.5-0.8B | 1 | 0 | 0.000000 | 0 | 0.000000 |
| Qwen3.5-0.8B | 2 | 0 | 0.000000 | 0 | 0.000000 |

Before execution, the runner required each policy to pass every model, batch
and pattern before any timing; both were excluded. The planned clock protocol
was2seconds warmup, own null before comparison,10ABBA rounds,3warm calls and
5samples/window, no filtering or normalization. Its later retention screen
required5% GPU and wall improvement in every profile, stable own-nulls within5%,
and a complete unchanged confirmation. None of these speed claims is measured.

## Costs and scope

The isolated projection sets add133,103,616bytes (126.9375MiB) of resident BF16
weight buffers, summed across three separately executed processes. The same
buffers are reused by B1/B2. This sum is not a single-process peak RSS or a
full-model extrapolation. Upload plus original preparation, additional layout
packing and CPU bit verification totaled487.0775ms across the numeric processes;
no matched no-extra-copy setup delta or transient/peak RSS was measured.

Windows preflight checked25 irregular/padded arbitrary-payload layouts and
204,800 padded BF16 words. The reproducible preparation helper gives identical
source hashes to the actual Mac generation. These CPU tail cases do not replace
a GPU ownership/abort/tail suite, which was not implemented. This is decoder
projection evidence only: no vision tower, sequential network, TTFT, decode
throughput or fresh vLLM comparison. The guard excluded competing runtimes and
probes during every executed stage; all original stages are terminal and archived.

## Reproduction and evidence

```text
python -X utf8 test2026/metal/prepare_block_packed_prefill.py REPOSITORY FRESH_OUTPUT
python3 launch_block_packed_prefill.py ROOT ROOT/logs/block-packed-prefill-20261016
python3 archive_block_packed_prefill.py ROOT ROOT/logs/block-packed-prefill-20261016 ARCHIVE
python -X utf8 test2026/metal/verify_block_packed_prefill.py RESULTS/initial VERIFICATION_JSON
```

[Raw sources, commands, checkpoints and failures](initial/evidence/manifest.json),
[Windows audit](initial/verification.json), [Mac audit](initial/mac-verification.json),
[Archive byte audit](initial/archive-verification.json),
[CPU-only preparation](preparation/preparation.json).

The50-file archive includes20 consumed source identities, all four generated
sources, exact defines/commands, stage exit and guard logs, raw numerical failures,
configuration files and size/hash identities of omitted checkpoints, raw slices
and executable. The Mac/Windows audits reproduce identical results. Archive:
`f1eaf86ceb751984013df8b723b2ff04f45d158cb8e04d04d3347baee78f0579`.
Accepted reference library:
`d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.
Environment inherits the verified M4/16GiB, macOS15.6.1, Apple clang17 and
Python3.14.6 host; no separate pre/post thermal or hardware-cache flush was taken.
