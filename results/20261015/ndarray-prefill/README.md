# Native MPS NDArray prefill investigation

Status: initial failure and two repaired trained cohorts terminal, archived and audited on Mac and Windows.
No serving defaults changed. No complete-model, TTFT or vLLM improvement yet.

Hypothesis: a deferred-transpose MPSNDArray product may reduce GPU prefill or
host encoding time relative to current MPSMatrix/custom BF16 projection paths.
Keep all original learned values, FP32 activations/results and original bounds.
Policy1 uses existing exact-half storage, three sources with alpha and a zero
addend. Policy2 expands original BF16 values exactly into a retained FP32 cache
and uses two sources with unit alpha. Policy3 is the native null control.
The FP32 cache adds storage; internal MPS packing/allocation is unmeasured.

Test all29 original first-layer projections across three models, B1/B2 physical
rows24/44,26/52,80/160, four original patterns. Failed numerical groups receive
no timings. Each passing group uses two ABBA rounds with eight samples per
position; null uses identical native GPU work. Retain original float-key path.
Stable input/output slots and a private bounded256-entry cache hold owners.
No synthetic ownership suite or full-model validation is supplied by this test.

Before repaired trained timing, declare a screening rule: repeat any candidate
policy only if at least one profile passes all original gates, shows >=5% lower
GPU and wall means, and its model/batch null GPU and wall means are within5%.
Promotion additionally requires a fresh confirmation under the same rule,
fresh complete native validation/full outputs and fresh matched vLLM timing.
This pragmatic screen is not a confidence interval or a model-level prediction.

## Initial native failure

PID77664 exited1 in12.306445375s. Compilation passed; the first text-model
execution stopped at `NDArray transposed view failed`, before numerical output.
Archive9d03762203595c7d8396fa93fa87ff82a91c6e7f9bf70295ebcf0602fb4b9933
contains52 files, preserved in [initial failure](initial-view-failure/evidence/manifest.json).
The API diagnostic established physical-versus-logical descriptor semantics;
see [separate compatibility evidence](../ndarray-matrix-api/README.md).
The repair queries the descriptor's logical dimensions; original bounds and
arithmetic stay intact. The original failure was archived before this change.

## Reproduction

On the verified Mac root `/Users/zsdx/garnet-metal-investigation`:

```text
python3 source/test2026/metal/launch_ndarray_prefill.py ROOT ROOT/logs/ndarray-prefill-repaired-20261015 component
python3 source/test2026/metal/archive_ndarray_prefill.py ROOT ARCHIVE ndarray-prefill-repaired-20261015
python3 source/test2026/metal/verify_ndarray_prefill.py EXTRACTED VERIFICATION.json
```

`ROOT`, `ARCHIVE` and `EXTRACTED` are placeholders. Launcher verifies prior
archived compatibility gates, waits for a quiet interval and monitors exclusivity.
The runner captures commands/versions/SDK headers/generated code/assets/guards.
No prefix, speculative, image or learned-input reuse. Init/first/warm model
claims require separate complete-model execution, not these component samples.

## Repaired trained execution and confirmation

PID78431 exited0 in79.905263500s; PID79830 exited0 in77.154079084s.
Each frozen cohort passes the source/command/asset/guard audit, with72 numerical
records and768 timed executions across12 admitted profiles. Six candidate
profiles fail the original native-comparison envelope on pattern1 (text B1,
coder B1/B2, both policies); they retain failures and have no timed windows.
All native controls and candidate FP64 envelopes pass. Numerical rejection is
not erased because a different reference bound passes.

Initial archive2675b4e32ddc1b3f0fab8c4930237ca3a25a8a45f14c4d8dfeebbaf2b33bd263;
confirmation d658c582f33541f953e225599af130275b7cb23af40c748ec8fbf25543daec35.
Both contain59 manifest files. Source/helper/generated code, flags, learned
assets, SDK, hardware and versions match exactly across cohorts. Full raw
samples and admitted/rejected groups are in [initial audit](repaired/verification.json)
and [confirmation audit](confirmation/verification.json).

Fractions below are candidate/control (lower is faster); mixed=policy1,
F32=policy2, null=policy3. The null has identical native GPU work in both arms.

| Model | Batch | Policy | Initial GPU | Initial wall | Confirm GPU | Confirm wall |
|---|---:|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1 | null | 0.996495 | 1.000230 | 0.989143 | 0.942136 |
| Qwen3-0.6B | 2 | mixed | 1.000456 | 0.967642 | 0.998624 | 1.028565 |
| Qwen3-0.6B | 2 | F32 | 0.997366 | 0.886752 | 0.998134 | 0.897025 |
| Qwen3-0.6B | 2 | null | 1.017166 | 1.036512 | 1.011081 | 1.010426 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | null | 0.996592 | 1.004339 | 0.980929 | 0.962975 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | null | 0.973128 | 0.975736 | 0.950881 | 0.955612 |
| Qwen3.5-0.8B | 1 | mixed | 0.950318 | 0.947973 | 0.981824 | 0.981563 |
| Qwen3.5-0.8B | 1 | F32 | 0.878001 | 0.823963 | 0.880955 | 0.833840 |
| Qwen3.5-0.8B | 1 | null | 1.022940 | 1.026699 | 0.965432 | 0.977348 |
| Qwen3.5-0.8B | 2 | mixed | 1.054250 | 1.043581 | 0.949636 | 0.946783 |
| Qwen3.5-0.8B | 2 | F32 | 0.963192 | 0.912920 | 0.926767 | 0.888284 |
| Qwen3.5-0.8B | 2 | null | 1.016735 | 1.014130 | 0.978040 | 0.981469 |

The predeclared screen selects only Qwen3.5 B1 F32: GPU savings12.20%/11.90%,
wall savings17.60%/16.62%, own null within5% in both cohorts. Other gains do
not satisfy the full rule; text B1 null wall also exceeds5% on confirmation.
[All selection observations](decision-analysis.json) retain those exclusions.
This is a component screen, not statistical confidence or end-to-end TTFT.

Proceed to a [confined native candidate](../ndarray-native/README.md) for full
validation and actual complete-model measurements. Restrict its execution to
row80, add bounded-cache native fallback and expose actual route counters.
Memory cost, first-use cost, final outputs, other-layer accuracy and full-model
speed remain unverified. Retained servingc186 and matched chart stay unchanged.
