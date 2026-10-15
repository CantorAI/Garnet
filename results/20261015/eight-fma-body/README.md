# Explicit-FMA eight-output decode: prepared locally, not executed

Hypothesis: explicit FP32 fused multiply-add may restore the original control
output bits in the lexical eight-output kernel. Both prior eight-output
variants passed the unchanged numerical envelopes but failed the mandatory
exact-bit gate; [their failures remain rejected](../coalesced-eight-body/README.md).
The native compiler's contraction/reduction behavior is not established.
This is a separate candidate, not a reinterpretation of those failed checks.

`make_eight_fma_body.py` replaces exactly four named accumulator expressions
with explicit vector FP32 `fma`. Original compact BF16 payload, learned values,
FP32 inputs/outputs, lane/loop order and control shader remain unchanged.
The context, header and harness are private copies. No serving source/default
changes. Context/header generation and Python syntax checks passed locally;
no Mac Objective-C++/Metal compile, GPU execution, correctness pass or timing
claim exists for this candidate.

All original29 trained projection matrices, B1/B2, four patterns,64/128/256
thread policies, independent original-checkpoint CPU packing proof and189
safety profiles remain required. Every numerical record still requires exact
native output bits plus the original FP64 gamma envelope and native comparison
bound. Only policies passing all four patterns may enter the original two
ABBA timing rounds; no failed sample is admitted. The new auditor regenerates
the lexical shader from the original four-output shader, then regenerates the
explicit-FMA transformation. Success would establish compatibility for those
fixtures, not explain the old compiler behavior or prove a full-model gain.

The existing command-buffer validation and native benchmark take precedence.
`launch_eight_fma_body.py` requires the exact controller13152 dispatch, native
terminal success, a preserved native archive and the72-complete-output audit,
then enforces its own quiet guard. This launcher has not been invoked. Future
GPU trials must remain sequential, and source bytes stay frozen until the
owned job is terminal and archived. Fresh component confirmation, unchanged
model gates, complete generation and repeated matched vLLM runs remain
necessary for promotion. Full goal unmet.

```text
# Prepared protocol only; actual invocation requires the upstream evidence.
python3 launch_eight_fma_body.py ROOT ROOT/logs/eight-fma-body-20261015 component
# After terminal capture:
python3 archive_eight_fma_body.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_eight_fma_body.py RESULTS RESULTS/verification.json
```
