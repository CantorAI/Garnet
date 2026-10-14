# Coalesced mixed-input matrix decode: rejected

The actual Apple M4 component run exited 0 in 42.38 seconds. Independent
verification covered 34 archived files, 64 matrix/batch checks, 49,280 validated
executions and 6,144 timed executions. The two exact-half weight exclusions
and three wide-exponent input rejections retain the original native fallback.
No full-model candidate was selected.

The experiment replaces the previous joined-components MPS product with
coalesced checkpoint-row loads, padded shared-memory transposition and an
explicit FP32 SIMD matrix accumulator. Original BF16 learned values and FP32
inputs are reconstructed exactly through admitted half representations. Split,
product and FP32 recombination are included in the timed command buffer.
Padding's hardware bank-conflict effect was a hypothesis, not a measured fact.

| Model | Batch | Candidate / native GPU time | Candidate / native wall time |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 2.9583 | 1.3412 |
| Qwen3-0.6B | 2 | 2.2424 | 1.3431 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 2.3974 | 1.8919 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1.8325 | 1.5647 |
| Qwen3.5-0.8B | 1 | 2.1738 | 1.8580 |
| Qwen3.5-0.8B | 2 | 1.8144 | 1.6140 |

These are unweighted pools of the tested trained large matrices, not model
latencies or tokens/sec. Lower is better; every pool regressed. The unchanged
FP64 error-bound worst fraction is 0.04831 and native-comparison bound fraction
0.23136. Numerical eligibility does not make this a performance improvement.
Production shaders and the selected global-alpha library remain unchanged.

Exact commands, source snapshots, shader and binary hashes, hardware/compiler
versions, guards, inputs metadata and all timing windows are in `evidence/`.
Archive SHA-256: `a55dc886b3f607967fa61f28f778b4e9713c977fc0d4ebf36b8fd0a3c98edc16`.

```text
python3 run_coalesced_activation_decode.py ROOT ROOT/logs/coalesced-activation-decode-20261015/jobs
python3 archive_coalesced_activation_decode.py ROOT ARCHIVE
python -X utf8 verify_coalesced_activation_decode.py RESULTS VERIFICATION_JSON
```

The full 2x decode / half-TTFT goal remains unmet. There is no new vLLM comparison.
