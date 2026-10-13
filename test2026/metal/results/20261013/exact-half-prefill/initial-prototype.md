# Initial exact scaled FP16 MPS prefill prototype

This is the first, slow-packing prototype, not the final selected build.
Library SHA-256: `5960662c1bed65e22e9db259f01d689964815208c021c1093ea062947c936770`.
`GARNET_METAL_EXACT_HALF_PREFILL=ON` packs body matrices up to 16 MiB into
FP16 storage using a per-row power-of-two scale and exhaustive per-value
bit-roundtrip checks. Any unrepresentable matrix falls back entirely. Raw
BF16 decode/head storage remains unchanged. Multi-token input/output and MPS
accumulation stay FP32, followed by inverse per-output-column rescaling.

All 11 native checks passed, including real layers/heads/vision and new odd
stride, signed-zero, per-row scaling and rejected-encoding fallback cases.
The sequential native ABBA comparison executed 48 text/coder requests and 24
actual image requests against `build-delta-split`. Every prompt, complete
output and finish flag matches. Four warm requests per profile/build are
averaged below; first-use and initialization records are separate. Filesystem
caches are unflushed. Native vision executes anew for each real frame.

| Model | Batch | Control TTFT, ms | Candidate TTFT, ms | Decode ratio |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 104.12 | 87.39 | 1.000 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 71.56 | 57.20 | 1.002 |
| Qwen3-0.6B | 2 | 175.21 | 152.13 | 0.993 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 99.54 | 77.32 | 0.993 |
| Qwen3.5-0.8B, image | 1 | 597.25 | 564.38 | 1.005 |
| Qwen3.5-0.8B, image | 2 | 854.22 | 796.15 | 0.998 |

## Startup and memory cost

Despite lower warm TTFT, the initial packing loop has a large startup penalty.
These averages cover two fresh processes per configuration. RSS is the
individual native text/coder child high water, not unique GPU allocation.

| Model | Batch | Control initialization, s | Candidate initialization, s | Control RSS, GiB | Candidate RSS, GiB |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 1.002 | 11.185 | 1.506 | 2.329 |
| Qwen3-0.6B | 2 | 1.012 | 11.219 | 1.549 | 2.374 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.793 | 9.070 | 1.264 | 1.932 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.788 | 8.982 | 1.263 | 1.932 |
| Qwen3.5-0.8B | 1 | 2.132 | 13.528 | not reported | not reported |
| Qwen3.5-0.8B | 2 | 2.145 | 13.440 | not reported | not reported |

Initialization is excluded from the warm TTFT column above. The prototype
uses floating-point conversion and accesses Metal buffer.contents inside
the per-weight loop. It is superseded by later integer-conversion and pointer
hoisting experiments; these initial costs must not be hidden by warm metrics.

All six actual serialized diagnostics preserve complete outputs and confirm
the mixed path executes: text 392 operations/profile, coder 336, VL 835.
They split submissions and perturb cache/dispatch behavior; their times are
not serving speed evidence. Commands/exits, source/build/model/input hashes,
actual outputs, guards, numerical logs, startup and warm summaries are saved.
Raw RGB/patch arrays and binaries remain private on the Mac. There is no new
vLLM comparison or 2x/half-TTFT completion claim for this initial prototype.
