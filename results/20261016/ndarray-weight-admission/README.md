# Full body-weight admission: one VL matrix needs the original FP32 fallback

Status: both CPU jobs executed successfully and archived. Windows evidence
audit passes. This is weight eligibility/provenance, not GPU performance or
complete-model validation. The overall 2x/half-TTFT goal remains unmet.

| Model | Body matrices | Common-scale exact half | Common or row-wise exact half | No exact half | Additional selective F32 payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 168 | 168 | 168 | 0 | 0 |
| Qwen2.5-Coder-0.5B-Instruct | 144 | 143 | 144 | 0 | 0 |
| Qwen3.5-0.8B | 186 | 182 | 185 | 1 | 25,165,824 bytes |

Scan all 498 original BF16 body matrices in the full real checkpoints. Exclude
the special dense-text key projection that already uses a different FP32 path.
Keep the current 16 MiB source-weight cap, aligned K, scale range +/-64 and
all exact-value encoding tests. Eligibility assumes the retained validated
half pipeline and successful allocation. It is not per-weight GPU routing
instrumentation. Reconcile predicted common-scale calls with 18 actual
baseline requests from the [completed native cohort](../../20261015/ndarray-native/README.md).
Every counter agrees with its original chunk plan.

Four VL matrices lack a common scale, but three admit row-wise exact half:
layer 0 linear-attention Z, layer 16 linear-attention QKV and layer 23
self-attention V. Coder layer 8 gate also admits row-wise half. Replacing
all non-common matrices would unnecessarily replace these existing routes.

The only unsupported matrix is VL
`model.language_model.layers.0.linear_attn.in_proj_qkv.weight`, 6144x1024.
Its raw BF16 SHA256 is
`8904cdb2c2e3aa5fc158d074e3b84fb5165a6637f2bd38f67fb93b29406fe0dc`.
Fourteen rows require scale 138, outside +/-64: 397,423,438,453,470,476,479,
2445,2471,2486,2501,2518,2524,2527. Their maximum BF16 exponent is 4;
these are tiny normal BF16 values, not BF16 subnormals. Preserve every value.
The [earlier expanded-scale GPU experiment](../../20261013/extended-half-scale/README.md)
already tested lossless wider scales and found no useful speed gain. Do not
repeat that unchanged method or loosen numerical bounds.

NumPy audits the complete checkpoints and verifies bitwise FP32 roundtrips
for admitted matrices. An independent Foundation-only C++ executable uses
the original frozen `exact_half_codec.h`, audits all five non-common matrices,
matches their SHA256 and flags, and checks **19,548,160** admitted values
bit-for-bit through native `_Float16` and `std::ldexp`. The 14 rejected rows
are excluded from that roundtrip count; they are not zeroed or admitted.

The Python job exits 0 in 63.683980790898204 s; original-C++ compile/audit job
exits 0 in 1.4974965420551598 s. Full commands, versions, source identities,
checkpoint headers, terminal statuses, source snapshots and reported counters
are retained under [evidence](evidence/manifest.json). Raw weights/checkpoints
and compiled binary are omitted by size/SHA identity. The Windows verifier
recomputes archive-member hashes, header coverage, counts, budgets, raw-output
counter reconciliation and C++ log consistency; it cannot rerun omitted weight
arithmetic without those original assets. This distinction is part of the
[verification scope](verification.json).

```text
python3 audit_ndarray_weight_admission.py ROOT OUTPUT
python3 run_ndarray_weight_admission_cpp.py ROOT OUTPUT_CPP
python3 archive_ndarray_weight_admission.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_ndarray_weight_admission.py ADMISSION_EVIDENCE NATIVE_COMPARISON_EVIDENCE AUDIT_JSON
```

Archive SHA256:
`66d403a090e6275e9a623459b33b9a9ffd30de19e9811cc3ce84a36da74f8e4e`;
24 verified files. New [selective component protocol](../ndarray-selective-prefill/README.md)
uses `!ExactHalfReady()`, retaining the 185 admitted VL routes. Projected
logical F32 payload is 24 MiB versus 1,990,459,392 bytes (1.854 GiB) for broad
expansion; neither value is an observed RSS measurement. No inference defaults
or retained library changed as a result of this CPU audit.
