# QV fusion with the original key projection

Final status: reject. All learned component numerical gates pass, including
bit-identical original K outputs, but no model meets the predeclared complete
endpoint timing rule. Do not repeat or integrate this unchanged variant.

Pack Q and V into an exact common-scale half weight matrix, encode one MPS
product, then scatter into the original independent Q/V outputs. K calls the
original EncodeBF16MatVec route. In the retained text/coder configuration,
floatPrefill selects split BF16 key projection with FP32 accumulation; it is
not an FP32-weight MPS projection. The archived factory comment describing an
original K MPS call is imprecise; the actual code calls the original function.
The runtime call and original key output bits are verified.

Keep prior triple eligibility and scale selection conservatively: K still
participates in CPU admission and scale choice, while its original GPU path
and storage remain unchanged. This is not a complete study of every pair
that could be admitted by an independent Q/V scale. All 58 previous triples
are covered; 56 prepare successfully and two retain original fallbacks.

## Executed primitive and learned checks

Primitive PID 42044 exits 0 in 11.290640791 s. All 24 changed-input dyadic cases,
six row counts, unequal widths, original key bits, pending/foreign ownership,
alias rejection, abort/recovery, fallbacks and bounded 64-group/2-reject cache
checks pass. Extra synthetic Q/V weight payload is 256 bytes, shared by shapes.
The 42-file archive is regenerated and audited on Mac and Windows.
[Primitive audit](primitive/verification.json).

Learned PID 42323 exits 0 in 83.629794833 s. All 116 profiles and 464 changed
structured-input cases pass, including 52 original FP32-accumulation key
matrices, 16 fallback cases, 177,553,408 reference-value checks and 48,758,784
scatter-bit checks. Candidate/control projection error ratio is zero on these
structured inputs; this does not prove equivalence on arbitrary activations.
The following original-K bit checks are also required per profile and verified:
**10,723,328 values** in the complete numerical stage.
[Complete learned audit](initial/verification.json).

## Predeclared timing rule and complete result

Before execution, fix first/last previously admitted layers, both batches,
2 s alternating warmup, own-null followed by comparison, 10 ABBA rounds,
3 warm calls and 5 measured samples per window. Retain all 4,800 samples.
A model requires at least 5% lower GPU and wall time at all four endpoints,
with each own-null within 5%. No model passes; no confirmation is admitted.
[Decision from all samples](initial/decision.json).

Fractions below are candidate/control; lower is better. Nulls are exclusion
checks only and never normalize candidate gains.

| Model | Layer | Rows | GPU fraction | Wall fraction | Null GPU fraction | Null wall fraction | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 26 | 0.984749 | 0.969081 | 1.037224 | 1.013537 | No |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 52 | 1.014640 | 0.926307 | 1.018487 | 1.004230 | No |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 26 | 0.933459 | 0.930381 | 1.004732 | 1.041865 | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 52 | 1.002376 | 0.928738 | 0.994362 | 0.986651 | No |
| Qwen3-0.6B | 0 | 24 | 1.063689 | 0.999854 | 1.019426 | 1.034210 | No |
| Qwen3-0.6B | 0 | 44 | 1.096286 | 1.038628 | 1.010703 | 1.005306 | No |
| Qwen3-0.6B | 27 | 24 | 1.071997 | 1.023511 | 0.989625 | 0.970101 | No |
| Qwen3-0.6B | 27 | 44 | 1.075359 | 1.067109 | 0.991048 | 1.001611 | No |
| Qwen3.5-0.8B | 3 | 80 | 1.288411 | 1.176937 | 0.987773 | 0.968817 | No |
| Qwen3.5-0.8B | 3 | 160 | 1.392082 | 1.262729 | 0.993461 | 0.990057 | No |
| Qwen3.5-0.8B | 19 | 80 | 1.265126 | 1.157771 | 1.008896 | 1.003753 | No |
| Qwen3.5-0.8B | 19 | 160 | 1.376343 | 1.243971 | 0.995975 | 0.997216 | No |

Text CPU encode time improves, but GPU time increases by approximately 6–10%
and wall time does not meet the gain rule. Coder has mixed endpoint results;
VL is slower. Fewer API encodes do not establish fewer GPU dispatches or an
end-to-end speed gain. The scatter and changed product shape remain costs;
this experiment does not isolate each cost causally.

## Preparation and storage costs

These are component measurements, one learned group at a time. Payload sums
estimate duplicate weights if all admitted groups were resident; do not sum
component preparation clocks into a measured model startup claim.

| Model | Admitted groups | Duplicate QV weights MiB | First shape preparation min ms | First shape preparation max ms |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct | 24 | 42.000 | 53.863292 | 55.618375 |
| Qwen3-0.6B | 28 | 168.000 | 195.198459 | 200.536792 |
| Qwen3.5-0.8B | 4 | 36.000 | 277.112834 | 281.483834 |

[Complete cost ranges and source identity](cost-summary.json).
The integer inverse helper is not used in this experiment.

Frozen archive SHA256 values:

- Primitive: `01d7c69ba0626d31fa5ce4113ad49218bdb405eda643152a6d41ecf11cbf8cb6`.
- Complete learned cohort: `2483d53e0765d9e096230aaec8553b64286ee40b2c46a9849cb80b7967e5d698`.

Both archives are independently extracted, regenerated and audited on Mac and
Windows. This variant has no native model timing, integration or fresh vLLM
result. A proposed output-view variant would remove the scatter but requires
separate offset/ownership/consumer evidence and cannot reuse these timings.
