# Dependent MLP with existing reference rounding: audited, six profiles excluded

Hypothesis: evaluate the same complete gate/up-SiLU-down workload using
the original trained reference's stage boundaries, correcting the previous
whole-chain FP64 reference mismatch. Four trained blocks remain: text/coder
layer 0 and VL layers 0/3. VL blocks are sequential separate graph calls.
Inputs, actual BF16 checkpoint bits expanded losslessly to FP32, FP32 graph
arithmetic, native library, placeholder/constant policies and all tolerance
coefficients are unchanged. This is a reference correction, not a speed
optimization or a repeat of an identical experiment.

The generated harness makes only three reference substitutions: gate/up dot
products accumulate wider then cast to FP32; activation executes in FP32;
the wider down dot casts to FP32. The verifier reconstructs these changes
and checks the frozen `reference_dense.py` semantics. Historical JSON names
`fp64_worst` and `control_fp64_worst` remain, but here mean error against this
stage-rounded reference, normalized by the unchanged 5e-6+5e-5*abs(reference)
bound. Values over 1 fail. Native comparison uses its separate unchanged bound.

PID52956 is terminal exit 0 (38.223373 s). Audit: 36 files, 48 numerical
records, 12 first graph calls and 384 admitted timed submissions. Six profiles
still fail the reference on high-amplitude alternating pattern 1 in BOTH
unchanged native and graph: text B1/B2 and coder B2, each weight policy.
Worst native/graph normalized reference errors are 2.420709/2.529337,
2.887737/3.140284 and 1.308438/1.286163 respectively. These profiles have no
timing windows. They remain numerical limitations; changing reference
rounding did not make all tests pass.

All 48 graph/native comparisons pass, worst ratio 0.473572. That relative
agreement is not a waiver of independent reference failures. Only coder B1
and VL profiles pass every component gate and are timed below:

| Model / batch | Weight policy | Native wall ms | Graph wall ms | Graph/native wall |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct / 1 | placeholder | 1.311486 | 1.206151 | 0.919683 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | constant | 1.324608 | 1.285091 | 0.970167 |
| Qwen3.5-0.8B / 1 | placeholder | 3.198436 | 2.836174 | 0.886738 |
| Qwen3.5-0.8B / 1 | constant | 3.677749 | 3.514564 | 0.955629 |
| Qwen3.5-0.8B / 2 | placeholder | 6.073017 | 5.066973 | 0.834342 |
| Qwen3.5-0.8B / 2 | constant | 4.885328 | 4.766169 | 0.975609 |

Each admitted profile uses two ABBA rounds, eight samples per position;
all sequential block work is inside wall time. First graph calls and setup
are separate. Memory and isolated compilation costs are unmeasured here.
The 2.4–16.6% lower admitted component wall means do not imply end-to-end
TTFT gains, decode gains, statistical significance or repeatability. Comparison
to the earlier reference study's wall means is not a controlled speed claim.

Decision: prepare an OFF-default native integration for the original full
trained-model checks, ownership/abort/recovery and complete-output comparison.
Keep all reference failures visible. The original model validation tolerances
must pass unchanged before any serving benchmark is admitted.

```text
python3 launch_stage_rounded_mlp_graph.py ROOT ROOT/logs/stage-rounded-mlp-graph-20261015 component
# After terminal status only:
python3 archive_stage_rounded_mlp_graph.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_stage_rounded_mlp_graph.py RESULTS RESULTS/verification.json
```

These are invocation templates; frozen executed commands and source identities
are under [evidence](evidence/manifest.json). `ROOT` denotes the isolated Mac
investigation root. Archive SHA-256:
`2d7fd927af4d211044b793d44dc05331497fe29720aa5a5c24f950c120d7e02a`.
Audit: [verification.json](verification.json). Full-model speed, RSS, first
request cost, complete outputs and a fresh vLLM cohort remain unmeasured for
this graph integration. The full goal remains unmet.
