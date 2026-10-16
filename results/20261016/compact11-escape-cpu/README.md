# Three-bit exponent tags plus escapes: descriptive CPU study

Status: actual CPU scan completed and archived; Mac and Windows evidence audits
pass. No packet encoder/GPU kernel/inference implementation or measured speed
gain. No serving change or goal target pass. Do not cite these byte fractions
as inference throughput, GPU traffic or process memory savings.

The retained [coalesced body codec](../../20261015/coalesced-body-block/README.md)
stores four output channels by128 input values:512 sign/mantissa bytes,
256 four-bit exponent-tag bytes, and four metadata bytes. Exponent spans above15
use raw BF16. Hypothesis: replace four exponent bits with three, reserve tag7
for an exact original exponent-byte escape, and preserve every original BF16
bit before unchanged FP32 arithmetic. This differs from changing output grouping,
threadgroup size, FP32 sum order or model precision.

Scan all29 original first-layer fixtures, verify full checkpoint hashes and each
raw checkpoint slice/fixture hash, and apply the retained large-body admission.
Twenty matrices qualify:6 text,3 coder,11 VL. Across129,984 blocks,125,525 have
the retained compact span and4,459 use raw fallback. For each compact block,
choose the seven-exponent window covering most values, with the lowest base
on ties, then record the exact number of escapes. The CPU reconstruction
expression preserves **64,268,800** entries bit-for-bit, including source zeros
and tiny values. This is not a packed-bit-stream roundtrip or GPU numerical test.
It does not describe all transformer layers or vocabulary-head weights.

| Model | Eligible matrices | Blocks | Retained modeled bytes | Ideal fraction | Ranked/aligned fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 6 | 28,672 | 22,389,248 | 0.938683 | 0.985795 |
| Qwen2.5-Coder-0.5B-Instruct | 3 | 25,536 | 19,940,096 | 0.936670 | 0.983627 |
| Qwen3.5-0.8B | 11 | 75,776 | 59,159,808 | 0.938102 | 0.985190 |

Fractions compare hypothetical packet bytes plus unchanged raw fallback data
and metadata, not actual memory transactions. The ideal estimate assumes a
free escape-rank/address lookup:704 tag/sign bytes plus E exponent escapes and
four metadata bytes. The ranked estimate adds32 bytes, one per SIMD lane, and
rounds each candidate payload to16-byte alignment; it retains the original
772-byte packet whenever that is smaller. Both assume four-byte metadata can
also encode the required variable packet addresses; that layout is unimplemented.
Original BF16 masters/raw fallback storage remain available. There is no RSS
or cache-effect measurement, and no CPU packing-cost measurement for a working
packet encoder. GPU lookup/branch/address work remains unknown.

The initial archived CPU fraction denominator omitted four common metadata bytes
for each raw-fallback block. Preserve that report unchanged. The separate
[corrected budget analysis](budget-analysis.json) adds that identical term to
both arms; the table uses the corrected values. Full histograms and original
ratios remain in the [raw report](evidence/logs/compact11-escape-cpu-20261016/report.json).

Decision after this descriptive study: do not select a GPU candidate from these
numbers. Ranked/aligned modeled savings are only1.42% text,1.64% coder,1.48% VL;
the ideal6.1--6.3% assumes unimplemented free lookup. This is a prioritization
decision after seeing CPU results, not a predeclared statistical speed gate or
proof that every possible11-bit implementation loses. No favorable timing
search or inference success is claimed. Revisit only with a concrete lookup
design and unchanged numerical/full-model/repeated-vLLM validation.

The CPU job exits0 in4.830290125s. It ran during native numerical validation,
with no native/vLLM performance cohort admitted during overlap. This CPU wall
time is not an inference benchmark. No Metal, MLX, model loading or inference
runtime is used by the analysis. Source snapshots, Python/NumPy versions,
checkpoint/fixture identities and terminal status are archived. Raw weights
are omitted by size/hash. Both-host auditors recompute byte budgets from the
captured histograms and verify provenance; they do not rerun omitted weights.
[Verification scope](verification.json), [archive manifest](evidence/manifest.json).

```text
python3 analyze_compact11_escape_cpu.py ROOT OUTPUT
python3 archive_compact11_escape_cpu.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_compact11_escape_cpu.py EXTRACTED_RESULTS AUDIT_JSON
python -X utf8 test2026/metal/analyze_compact11_escape_budgets.py EXTRACTED_RESULTS
```

Archive11 verified files, SHA256:
`94fba9ef5ba375c688e574b079a04fddf0ace612c71224c130c17c869a0af4d3`.
