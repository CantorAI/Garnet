# Complete dependent MLP graph: audited stress experiment

Owned PID52062 waits for exact core graph PID50246 to be terminal, then a
five-second quiet interval. It compiled successfully and is running the three
models sequentially. No competing inference or GPU component may overlap.

The workload preserves a real MLP dependency chain: two gate/up projections,
(gate / (1 + exp(-gate))) * up, then the down projection. It uses checkpoint
MLPs at text/coder layer0 and VL layers0/3 (four blocks,12 trained matrices).
VL blocks use separate graph objects and sequential calls; no operations are
fused across actual attention/layer boundaries. Each block uses one native
submission or one synchronous graph call. Set wall time includes every block.

Native execution uses the current retained c186 library and its existing
BF16-storage/FP32-arithmetic MLP operations. Graph tensors and learned values
are FP32, with lossless expansion of original BF16 weights. Policy1 supplies
weight placeholders; policy2 embeds constants. Gate/up share the actual MLP
input, and graph intermediates follow the original dependency chain.
Only final outputs are graph targets, allowing legitimate internal fusion.

Four original input patterns are tested at physical rows24/44,26/52,80/160.
CPU FP64 computes the complete gate/up, nonlinear activation and down result.
Both native and graph final outputs must meet the existing trained-reference
envelope5e-6+5e-5*abs(reference); graph/native differences must meet the same
unchanged native-comparison envelope. This compound reference differs from
the earlier isolated-matmul gamma bound; it is not a widened tolerance.
Any failed native or graph profile has no timing windows. No failed input is
removed or adjusted to obtain a performance result.

Expected coverage is48 aggregate checks,12 first graph-set calls and up to12
admitted timing profiles with768 submissions. These are expected counts,
not audited results yet. Every admitted profile uses two ABBA rounds and
eight samples/position after common warming. First/setup costs are separate.

This remains an isolated MLP test. It does not include a full transformer
layer, attention, norm/residual operations, pending-input native/graph command
interoperability, complete generated outputs, model TTFT or a vLLM result.
Integration still requires ownership/abort/recovery, trained-model validation,
full-model repeats and a fresh matched comparator. The full goal is unmet.

```text
python3 launch_dependent_mlp_graph.py ROOT ROOT/logs/dependent-mlp-graph-20261015 component
# Only after the exact job is terminal:
python3 archive_dependent_mlp_graph.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_dependent_mlp_graph.py RESULTS RESULTS/verification.json
```

## Audited outcome and reference correction — 2026-10-15

The launch-time protocol above is preserved as history. PID52062 is terminal
exit 0 (38.668088 s). Independent verification covers 30 archived files,
48 numerical records, 12 first graph calls and 256 admitted timed submissions.
Eight text/coder model/batch/policy profiles fail the compound CPU reference
on pattern 1 in BOTH the unchanged native control and graph candidate. These
profiles have no timing windows. All 48 graph/native comparisons pass the
unchanged comparison envelope. Process exit 0 does not mean all gates passed.

Reference-design correction: the initial whole-chain FP64 computation does
not reproduce the existing trained reference's FP32 rounding after each
projection and activation. Its bounds had been described as the existing
trained-reference test too broadly. Preserve this experiment as a stress
test; it cannot establish equivalence to that existing reference. The
[stage-rounded follow-up](../stage-rounded-mlp-graph/README.md) changes only
CPU reference rounding. It is a distinct experiment, not a replicate, and
does not erase these failures or relax tolerance coefficients.

Only VL profiles were admitted to timing in this experiment:

| Model / batch | Weight policy | Native wall ms | Graph wall ms | Graph/native wall |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-0.8B / 1 | placeholder | 3.780941 | 3.574172 | 0.945313 |
| Qwen3.5-0.8B / 1 | constant | 4.028665 | 3.421134 | 0.849198 |
| Qwen3.5-0.8B / 2 | placeholder | 5.647932 | 5.210079 | 0.922476 |
| Qwen3.5-0.8B / 2 | constant | 4.791042 | 4.255926 | 0.888309 |

These are component wall means, not model TTFT, decode throughput or an
independent repeat. The first-call/setup records remain separate. Peak RSS
and isolated graph compilation time were not measured in this component
study. No production or whole-model claim follows.

Archive SHA-256:
`b1254e4b96986cf214deff72813ebd5eb6d665dda2a85457b6a4fd7c9120b6f2`.
Raw files and identities: [evidence manifest](evidence/manifest.json).
Audit: [verification.json](verification.json). The full goal remains unmet.
