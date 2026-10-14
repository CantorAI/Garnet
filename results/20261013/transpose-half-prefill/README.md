# MPS exact-half weight layout: no broad serving promotion

This component experiment stores the same exact scaled-half encoded weight
bits in K-by-N layout and uses `transposeRight:NO`, against the retained
N-by-K / `transposeRight:YES` MPS path. FP32 activations, accumulation,
outputs, column scales and rescaling remain. There is no model, precision
or learned-value change. No serving source or selected library was modified.

Two actual sequential M4 / 16 GiB runs pass 432 checks in total. Each run
covers real first-layer matrices for all three checkpoints, irregular
synthetic dimensions, padded output strides and rows 22/24/26/32/44/52/64/80.
All compared finite output values are numerically identical (maximum
absolute difference zero); encoded weights are checked bit-for-bit across
layouts. This is not a claim about signed-zero output bit identity.
The fixed output bound remains 5e-6 + 5e-5*abs(control). No tolerance is widened.
Each timing uses two ABBA rounds, three warm-ups and twelve samples per
position. Timings include MPS multiplication and identical GPU rescaling.

| Model / physical rows | Initial median MLP GPU speedup | Repeat median MLP GPU speedup |
| --- | ---: | ---: |
| Text / 22 | 1.040 | 0.561 |
| Text / 44 | 1.038 | 1.036 |
| Coder / 26 | 1.047 | 0.756 |
| Coder / 52 | 1.007 | 1.037 |
| VL / 80 | 1.012 | 1.043 |
| VL / 64 | 1.034 | 1.035 |

Text/coder batch-1 gains do not repeat; important matrices regress.
The small VL component signal does not establish a full-model gain and
cannot justify broadly replacing the retained path. The experiment is not
promoted to serving. No model decode rate, TTFT or vLLM speedup is claimed.

Each run explicitly excludes the first-layer VL QKV matrix because the
existing exact-half converter rejects it. Unsupported weights are not
rounded or silently admitted. A subsequent read-only admission audit covers
all 18 VL linear-attention QKV matrices: 17 meet the current integer rules.
In the first matrix, 14 of 6144 rows have extremely small BF16 weights;
14336 values lie in rows rejected by the current scale limit. Their largest
magnitude is 1.17549435e-37 and the ideal row scale reaches 138, beyond the
current +/-64 limit. Clamping that scale does not make the matrix admissible.
This diagnosis suggests testing exact extended-scale reconstruction on GPU;
it is not evidence that such a path works or improves inference.

The initial admission diagnostic incorrectly asserted that every row's
largest value satisfied the existing scale rule. It failed before writing
a report. Corrected reports record rejected rows explicitly. The public
audit reran successfully, saves its own/checkpoint hashes, and reports only
CPU integer eligibility; GPU reconstruction remains untested.

## Actual commands and evidence

```text
python3 run_transpose_half_prefill_probe.py INVESTIGATION_ROOT OUTPUT_DIR
python3 audit_exact_half_qkv_admission.py CHECKPOINT OUTPUT_DIR
```

Both GPU jobs have terminal exit code 0 and ran after the refreshed
comparator suite completed. Exact compiler/runtime commands, generated
asset identities, source hashes, raw ABBA windows, checks, exclusions and
0.5-second process monitors are under `evidence/`. Matrices and binaries are
omitted from Git with sizes/hashes retained. This remains a component study
with deterministic sinusoidal inputs, not model hidden-state validation.
The full three-model 2x decode / half-TTFT goal remains active and unmet.

Archive SHA256:
`9f9c05fab337bffeed43160e7ad4ae7d9dce571e7c2e012fe16d4bbf7aee8324`.
