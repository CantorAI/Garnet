# Extended exact-half QKV scales: correct reconstruction, no useful speed gain

The retained codec rejects one of 18 Qwen3.5 linear-attention QKV matrices.
Its first-layer matrix contains 14 extremely tiny rows whose ideal scale is
138, beyond the existing +/-64 range. This experiment tests admitting those
learned values exactly, using an isolated codec range -112 through 141.
Production admission and the selected native library remain unchanged.
No weight is rounded or replaced with zero.

The first actual GPU check reconstructs every finite normal BF16 bit pattern,
plus signed zeros, using scaled half storage and `ldexp`: 65,026 values per
precise/fast-math compiler mode, zero bit mismatches. BF16 subnormal weights,
infinities and NaNs are excluded from this codec admission.

The actual first-layer QKV projection then passes the existing output bound
5e-6 + 5e-5*abs(FP32 control), but fails a deliberately stricter diagnostic
for the tiny columns. Initial and diagnostic jobs stop at 22 rows with exit
1. The independent FP64 dot reference identifies large relative errors in
the unscaled FP32 MPS control too: its intermediate terms are extremely
small. Naive half rescaling is much closer, but loses a subnormal output.
The diagnosis is consistent with intermediate/result underflow; it is not
an observation of undocumented hardware instructions.

An integer IEEE power-of-two rescaler preserves subnormal result bits,
signed zero and nearest-even underflow rounding. A separate GPU test covers
all finite BF16 patterns, including source subnormals, and FP32 exponent/
mantissa edge fixtures over 13 positive/negative scales. Its 888,420 cases
per compiler mode match host `std::ldexp` bit-for-bit, including overflow
results. This is a sampled FP32 test, not exhaustive over all 2^32 words.
It does not add BF16 subnormal weight admission to the serving codec.

Two actual projection variants use this helper: all output columns, and a
hybrid that uses it only when the column scale exceeds 64. Each passes eight
physical row shapes (22/24/26/32/44/52/64/80), output padding, unchanged
general output bounds, and the independent FP64 forward-error bound for
the 14 tiny columns. The bound is gamma(1024)*sum(abs(products)) + 1e-45,
with unit roundoff 2^-24. Agreement with the inaccurate tiny FP32 control
is retained as diagnostic data, not treated as a gold reference. Neither
the original general bound nor the independent forward-error bound is
widened to admit a candidate.

Both corrected variants have worst tiny-column forward-error ratio
0.00237020186, well within that independent bound.

The initial naive/control diagnostic failure remains in the archive. It
must not be mistaken for a failed general projection bound, a model output
change, or proof that the bit-rescaled candidate is faster.

| Physical rows | All-bit rescale component speedup | Hybrid component speedup |
| --- | ---: | ---: |
| 22 | 0.824 | 0.897 |
| 24 | 0.890 | 0.911 |
| 26 | 0.862 | 0.875 |
| 32 | 0.803 | 0.832 |
| 44 | 0.956 | 0.839 |
| 52 | 0.893 | 0.898 |
| 64 | 0.857 | 0.808 |
| 80 | 0.777 | 1.003 |

Ratios are FP32 MPS control / half-plus-rescale candidate GPU time. Each
variant uses two ABBA rounds, three warmups and twelve measured samples per
window. The candidate includes rescaling; the control writes directly into
the output and is not burdened with an artificial copy. All-bit rescaling
regresses every shape; hybrid regresses most shapes and is flat at 80 rows.
The variants are separate algorithms, not repeat confirmations. There is no
useful signal warranting a serving integration or full-model speed claim.

These are component tests with sinusoidal FP32 inputs and real checkpoint
weights, not model hidden-state or complete generated-output validation.
The selected serving code is untouched. The broader three-model 2x decode /
half-TTFT objective remains active and unmet.

## Commands and evidence

```text
python3 run_extended_half_scale_gpu.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_extended_half_projection.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_scale_bits_gpu.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_extended_half_bits_projection.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_extended_half_hybrid_projection.py INVESTIGATION_ROOT OUTPUT_DIR
python3 archive_extended_half_scale.py INVESTIGATION_ROOT ARCHIVE_PATH
```

The six executed jobs are terminal: codec/bit-scaler/all-bit/hybrid exit 0;
naive projection and its diagnostic exit 1 at the explicit tiny-output gate.
`evidence/completed/` retains exact commands, compiler/macOS/Python versions,
source/checkpoint hashes, raw timing windows, output diagnostics, monitors,
terminal statuses and hashes/sizes of omitted matrices and binaries.
`source-initial/` preserves the pre-diagnostic source that matches the first
projection attempt's recorded hash. Only owned probe groups could be stopped.
Existing inference guards plus 0.5-second process samples reject competing
runtimes; arbitrarily short jobs between samples cannot be ruled out.
GPT-OSS, Pi and the Mac original checkout were not changed.

Full archive SHA256:
`ce0de60a0c3d133404cf51f56f1663da4108978524dc84fd91550e70cb061c7b`.
