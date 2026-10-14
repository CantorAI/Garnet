# Metal optimization decisions and comparison limits

This index records why methods were selected, the evidence behind each
decision, and the scope of the conclusions. Individual reports retain
commands, source/library/checkpoint hashes, versions, numerical checks,
complete outputs, timing samples, monitors, failed attempts and rejection
reasons. Component speedups are not full-model speedups.

| Method | Reason to try | Decision and evidence |
| --- | --- | --- |
| Session-owned, single-slot KV updates | Avoid copying the full attention cache at each decode step | Retained opt-in. Text decode improvement repeated: 7.4–7.7% at B1, 13.6–14.4% at B2. Coder/VL gains not established. [Report](resident-kv-inplace/README.md) |
| Exact dense prefill tails | Remove padded computation for the frozen 22/26-token prompts | Retained opt-in for supported small dense models. Repeated TTFT reductions: text B2 5.84%, coder B1 2.45%, coder B2 7.18%; decode effectively unchanged. [Report](exact-dense-tails/README.md) |
| Dual-accumulator vocabulary head | Reduce dependent accumulation in the large output projection | Rejected. A 7.3–8.6% component signal did not improve actual affected text B1 decode (ratio 0.99867). Production files restored. [Report](dual-text-head/README.md) |
| Transposed exact-half MPS weights | Improve MPS prefill access without altering encoded values | Not promoted. Component output checks passed, but text/coder B1 gains did not repeat. [Report](transpose-half-prefill/README.md) |
| Extended exact-half scales for first VL QKV | Admit the sole excluded linear-attention QKV matrix without rounding its tiny rows | Experimental only. Codec is exact; naive rescaling loses a subnormal result. Bit-based reconstruction passes targeted checks but adds cost; hybrid reconstruction regresses most shapes and is flat at 80 rows. No serving promotion. [Report](extended-half-scale/README.md) |

Earlier retained prefill/recurrence methods and rejected alternatives have
their own reports in this directory. This table is a current decision index,
not an exhaustive list of all historical attempts or an attribution of the
entire advantage over vLLM-Metal.

The [latest actual comparator report](retained-prefill-vllm/README.md)
records six warm profiles and 72 complete matched output sequences. B1
decode ratios are 1.510–1.846; B2 ratios are 2.289–2.610. Every TTFT fraction
is above 0.5. The all-profile 2x decode / half-TTFT goal remains unmet.

These are additional matched-FP32 measurements, with identical learned
values, prompts, image bytes, capacities, caps and stopping rules. Garnet
retains compact BF16 weight storage and uses FP32 arithmetic; vLLM-Metal
loads the losslessly expanded FP32 weights. Learned-value and arithmetic
equivalence do not mean identical memory traffic or weight-storage width.
The storage difference can contribute to a bandwidth advantage; this study
does not separately quantify that contribution or establish stock BF16 wins.

Other limits: small models (0.5/0.6/0.8B), batches 1/2, frozen short prompts,
one fixed VL image, greedy output, four warm requests/runtime/profile, and
equal rows finishing together at B2. Long-context workloads, diverse batches,
general concurrent scheduling, larger checkpoints and quality across a broad
task set are not established by these results. Prefix/encoder reuse and
speculation are disabled; runtimes execute sequentially. Terminal predictions
count in decode throughput; prefill and the first prediction do not.
Initialization/first-use are separate, and OS/driver caches were not flushed.
