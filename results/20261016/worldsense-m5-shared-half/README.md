# WS-M5-03: immutable shared exact-half body storage — rejected

This native prototype shares one exact scaled-half allocation between MPS
prefill and strict FP32 decode. It raises the per-matrix admission bound from
16 to 64 MiB, releases the redundant original BF16 allocation for admitted
matrices, and retains the original fallback when exact encoding fails. Normal
half words are converted exactly and corrected by integer FP32 exponent bits;
half subnormals use integer normalization, preserving signed zero. The existing
FP32 dot/reduction order and bounded body specialization remain. Original
learned values are preserved; this is not quantization.

The prototype passed native tests but **failed its WorldSense performance
screen and was not promoted**. More prefill coverage is not itself a useful
whole-model optimization when reconstruction slows generation.

## Declaration, failures and consumed builds

`protocol.json` precedes timing. It requires exact complete outputs, ≥20% 4B
TTFT/≥10% total improvement, decode preservation within 5% on either model,
and text TTFT/total preservation within 5%. These architecture-tradeoff limits
are distinct from the earlier rejected head and limited-half screens. Extra
integer instructions, startup packing and implementation complexity are
declared costs. No standalone inference timing supports the decision.

The first source copy omitted the unchanged nlohmann header; its failed
compile is preserved before dependency repair. Before timing, the normal-value
conversion was refined from an integer-only inverse to an exact normal-half
conversion plus integer exponent correction. Both variants pass exhaustive
GPU correctness checks. No isolated performance probe selected between them.

A first native-service attempt stalled opening the isolated serving source
under Documents, before any model allocation or candidate samples. The process
sample shows the import filesystem-open stack; macOS folder authorization is
suspected, not proven as the sole cause. The attempt's completed control,
logs and consumed library are retained in `documents-root-attempt/`. Identical
sources were packaged under `CantorAI/out/research`; both control and candidate
were rebuilt from that same source/toolchain, with only the three storage/
prefill flags differing. A fresh full comparison starts after that repair.
No favorable timing was selected by excluding a measured candidate failure.

The M5 compatibility probe passes FP32 control but aborts the tested BF16
MPSMatrix combinations, as on the older host. A header enum is not runtime
support proof. This capability probe is not an inference benchmark and runs
outside all controlled timing windows.

## Correctness and storage accounting

Three native CTests pass. GPU inverse reconstruction matches **8,189,952**
finite-half/scale combinations bit-for-bit across scales −64..64. Twenty-four
native decode cases cover widths up to 9216 and real 1.7B/4B projection shapes,
generic/specialized routes, B1/B2, FP32 bits and the original FP64 envelope.
Tests also cover one-buffer accounting, half subnormals/signed zero,
unrepresentable fallback, independent contexts and abort recovery. Executed
test source is saved separately from an additional prefill extension that was
prepared later but not executed after this method failed its performance gate.

A full 4B checkpoint admission audit uploads each of 248 decoder-layer BF16
matrices individually. **247 matrices / 99.706% of their bytes** use the shared
layout; one matrix keeps the original representation. Original layer payload
is 7,138,181,120 bytes; retained matrix/scale buffers total 7,142,699,008 bytes:
**4,517,888 bytes (4.309 MiB) extra**, not a second full body copy. This is
logical per-layer buffer accounting, not whole-model peak RSS, GPU working set
or a memory saving relative to the original deployed BF16-only model. Head,
vision weights, mappings, activations and private state are excluded.

One separate fixed car-image request, 16 output tokens and capacity512,
produces matching tokens and finite states across all32 layers, but states
are **not bit-identical**. Maximum absolute state difference is
0.0001294613. The diagnostic's worst pointwise dense-coefficient ratio is
8.37; do not call it a pass of the original deep numerical suite. Changed MPS
reduction ordering prevents a bit-equivalence claim even when decoded values
and sampled text match. Full state hashes, per-part error/norm values and the
correctness-only driver are retained. Raw FP32 snapshots remain in the local
work directory; their exact hashes and reproduction inputs are preserved.

## WorldSense result

Four fresh sequential control/candidate/candidate/control processes execute
the same fixed-frame current-Skill/search-only cases and text search planning
through WorldSense → Galaxy → native Garnet. All72 complete model-call token
sequences/texts and input identities match. Model/Skill/image budgets and
chunk16 are unchanged. Warm medians use the two later repeats of every case;
all initial warmup requests remain in raw JSON. Live bindings are restored
after each suite; benchmark events and states are not committed.

| Process | 4B total p50 (s) | 4B TTFT p50 (s) | 4B decode p50 (tokens/s) |
|---|---:|---:|---:|
| control-1 | 12.687 | 3.879 | 13.340 |
| candidate-1 | 15.589 | 3.449 | 9.694 |
| candidate-2 | 16.146 | 3.619 | 9.405 |
| control-2 | 15.116 | 5.412 | 11.878 |

4B TTFT improves11.09% and33.13%, but decode falls27.33% and20.82%; total
rises22.87% and6.81%. Text decode falls26.52% and25.33%. **Reject.** Control
drift is retained, not removed; these trials do not isolate every causal
contribution of reconstruction, scheduling, clocks or system pressure. The
original production library/profile is restored. No unchanged favorable retry
overturns the failed gate.

The earlier fixed-frame protocol's limits apply: this excludes video decode,
candidate scheduling, temporal memory, event persistence and search retrieval/
ranking. Model timings exclude loading, queues and retries. Estimated decode
uses `(tokens−1)/(total−TTFT)` and can include EOS work. Native/HTTP wall times
and VM/swap samples are separate; post-run RSS is not peak unified GPU memory.
The original full historical multi-profile numerical suite is not claimed as
a fresh M5/4B pass. All source/build identities, failures and raw observations
are retained for paper preparation rather than a general speedup claim.
