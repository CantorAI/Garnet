# Separate Q/K/V exact union: full learned CPU eligibility

CPU prerequisite completed and audited on Mac and Windows. No GPU arithmetic,
numerical-envelope, model-latency or decode speed claim follows.

Hypothesis: Garnet lowers separate dense attention Q, K and V projections
consecutively from the same input. Their unequal output widths can share one
prepared exact common-scale half weight buffer and one F32-input MPS product,
followed by a scatter to the three existing outputs. This is a distinct
operator from the rejected gate/up activation-only experiment. It may reduce
two product encodes but adds joined scratch, output traffic and preparation.
Internal MPS GPU kernel counts are not asserted.

The audit includes52 key matrices retained in FP32 storage by current text/coder
policy. An exact weight roundtrip alone does not prove that changing their
MPS route preserves original numerical envelopes or full-model outputs.
Those require subsequent GPU checks before any timing or native integration.

| Model | Separate Q/K/V triples | Exact common union | Existing combined linear layers |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 28 | 28 | 0 |
| Qwen2.5-Coder-0.5B-Instruct | 24 | 24 | 0 |
| Qwen3.5-0.8B | 6 | 4 | 18 |

Actual learned output widths Q/K/V are2048/1024/1024 for text,
896/128/128 for coder and4096/512/512 for VL dense attention. All share the
respective model input width. VL dense layers7 and23 each contain one value
rejected at the common scale; their entire triples require original fallback.
The18 already combined VL linear-attention layers are outside this separate
projection hypothesis and remain untouched.

Actual CPU PID6423 exits0 in12.951632625 seconds, including NumPy scans,
compilation and the original C++ codec. The C++ checker is generated from the
previous frozen pair checker by changing only arity2->3, counts76/152->58/174,
and pair/group names. It retains the original exact codec, raw learned-slice
SHA checks and every admitted value's original FP32-bit roundtrip.
Across174 matrices,173,670,398 values restore exactly and2 reject. No learned
value is clipped, rounded away or omitted to admit a group.

If independent original weights remain allocated, extra common-union payload
for all admitted groups would be224 MiB text,47.25 MiB coder and40 MiB VL.
These are logical estimates, not a GPU allocation or full-model RSS measurement.
No original weight cache is mutated or serving source integrated.

Freeze20 files, archive SHA256
`3142474829315dd54bbc8bb84ffa3b7d56ec0299c16c5e581d473afd1ca9c825`.
SCP exits0 before Windows extraction. Both-host auditors regenerate the exact
C++ transformation, check full learned shape/offset/hash coverage, baseline
FP32-key flags, original codec, compile-only Foundation dependency and every
captured eligibility/roundtrip count. Raw checkpoints are omitted by complete
hash; the Windows provenance auditor does not rerun their arithmetic.
[Frozen sources/commands](evidence/manifest.json),
[independent provenance audit](verification.json),
[complete CPU report](evidence/logs/qkv-union-half-admission-20261016/report.json).

Executed audit (ROOT=`/Users/zsdx/garnet-metal-investigation`):

```text
python3 ROOT/source/test2026/metal/audit_qkv_union_half_admission.py ROOT ROOT/logs/qkv-union-half-admission-20261016
python3 ROOT/source/test2026/metal/archive_qkv_union_half_admission.py ROOT ROOT/qkv-union-half-admission-20261016.tar.gz
python3 ROOT/source/test2026/metal/verify_qkv_union_half_admission.py EXTRACT EXTRACT/verification.json
```

Next: private prepared-context primitive with ownership/alias/pending/abort
controls; all58 triples at both serving physical row counts and changed inputs,
including actual original FP32-key baselines and mandatory fallback groups.
Retain original gamma/sumabs and5e-6+5e-5 comparison bounds. Declare timing
selection and sample design before GPU performance execution. No unchanged
repeat of the rejected MLP version, relaxed envelope or old-vLLM/new-native
ratio can establish the overall goal.
