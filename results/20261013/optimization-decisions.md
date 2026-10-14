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
| Native BF16 body loads / vector dots | Test compiler-generated conversion and shorter accumulation dependency without weight repacking | Default-OFF experiment retained, primary build unchanged. All validation and 120 complete requests pass. Repeated text/VL decode gains are 0.5–2.8%, with slightly worse confirmation TTFT; coder flat. [Report](native-bfloat-body/README.md) |
| Independent MPS pairs / joined gate-up prefill | Reduce independent projection latency by overlap or a single packed MPS operation | Shared-event scheduling rejected after dependency overhead regressed all shapes. Joined packing is bit-exact in two component runs; text 44-row and coder 26-row signals merit a bounded model test, while text 24/coder 52/VL 80 do not. No serving change or model speed claim. [Report](mps-pair-probes/README.md) |
| Bounded joined MPS prefill integration | Test the two repeated component shapes in actual trained layers and complete inference | Default-OFF experiment retained. All checks and 96 complete requests pass; warm TTFT falls 2.68% for text B2 and 2.14% for coder B1 in confirmation, with higher RSS and substantially worse first-use TTFT. Primary serving library unchanged. [Report](joined-prefill/README.md) |
| Common frame RGB copy elision | Remove redundant full-image copies before both inference runtimes | Retained exact-input CPU simplification, no added production dependency. Preparation saves a few ms in two scans; 48 matched native/vLLM requests do not establish a useful end-to-end TTFT gain. OpenCV route not selected; B1 decode/half-TTFT targets remain unmet. [Report](frame-frontend/README.md) |
| FP32 vision softmax/value fusion | Remove the probability output matrix and following value-contraction dispatch | Default-OFF experiment retained. Two component runs are bit-exact and 12–19% faster at serving shapes; all 18 handoff stages and 48 complete frame outputs pass. B2 warm TTFT reduction repeats at 1.24–1.43%; B1's initial 2.61% becomes 0.22% in confirmation. No decode improvement or new vLLM claim; primary library unchanged. [Report](vision-fused-context/README.md) |
| GPU-verified contiguous attention slots | Replace repeated latest-writer scans when actual prompt controls form a consecutive active prefix | Optional candidate retained. 1,008 component comparisons, 16 validation stages and 144 clean model outputs pass. Warm TTFT gains repeat for text/coder B1/B2 and VL B1 (confirmation 3–7%); VL B2 remains inconclusive and uses the control in the fresh comparator. No established decode improvement. Disk-full timing is discarded; 15.17 GiB of verified derived matrix copies were cleaned with checkpoint/source/evidence preserved. [Report](attention-prefix/README.md), [fresh comparator](selected-attention-vllm/README.md) |
| Direct resident-state zeroing | Remove per-state temporary CPU allocations and copies during request reset | Default-OFF option retained with a tradeoff. All 19 validation stages and 144 complete native outputs pass. Repeated text TTFT reductions are 4.8–8.9% B1 and 13.1–16.2% B2; text B2 decode is 1.7–2.4% slower. Coder/VL signals are small or mixed, first use is mixed, and peak RSS savings are unproven. No uniform serving promotion or new vLLM claim. [Report and commands](direct-state-zero/README.md) |
| Synchronous GPU buffer-fill reset | Test GPU clearing after replacing temporary CPU allocations | Rejected for integration. Two component conditions (CPU warmed and after GPU writes) pass 126 zero/padding checks, but GPU fill is slower in all six model/batch shapes. No serving code change; exact commands, failed compile and both successful runs retained. [Report](state-zero-gpu/README.md) |
| Sixteen-lane BF16 decode groups | Share SIMD work between two output groups while retaining learned bits and FP32 math | Broad integration rejected and five serving/worker files restored on both hosts. Two component runs pass 272 reference checks with modest gains, and all 19 native stages / 72 outputs pass, but full-model decode is 0.4–5.0% slower across all six profiles. No new vLLM claim; exact tested sources and restoration identities retained. [Report](decode-subsimd/README.md) |
| Dedicated MPS matrix-vector decode | Check a vector-specialized alternative to rejected mixed-input matrix multiplication | Mixed Float16 matrix / FP32 vector aborts on the M4 with a same-input-type assertion. The valid lossless-FP32 route passes 68 real-matrix checks but pooled GPU time is about 2x/4x slower for B1/B2. Rejected for integration; no serving change or full-model gain claimed. [Report](mps-gemv/README.md) |
| Threaded fresh WebP preparation | Reduce the measured CPU decode cost without changing image/model inputs | Optional path retained; default frontend unchanged. Two CPU scans save about 9 ms with exact RGB/patch bytes; 956 compatibility checks and 96 complete outputs pass. Native warm TTFT reductions repeat at 2.6–5.0% B1 and 1.8–2.1% B2. Decode is unchanged and relative vLLM-ratio changes are mixed; full joint targets remain unmet. Both dtype audits and terminal runs are retained. [Report](webp-thread/README.md) |

Earlier retained prefill/recurrence methods and rejected alternatives have
their own reports in this directory. This table is a current decision index,
not an exhaustive list of all historical attempts or an attribution of the
entire advantage over vLLM-Metal.

The [latest actual comparator report](selected-attention-vllm/README.md)
records six warm profiles and 72 complete matched output sequences. B1
decode ratios are 1.527–1.824; B2 ratios are 2.075–2.552. Every TTFT fraction
is above 0.5. The all-profile 2x decode / half-TTFT goal remains unmet.
Batch-1 decode needs another 9.6–30.9% speedup; all-profile TTFT needs
another 18.6–47.9% reduction. Garnet's setup is faster, but its first request
after setup is slower in all six profiles; these intervals are reported
separately from warm latency.

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
