# Repeated native inference with original-order vision prefetch

The optional vision prefetch kernel reduces image-to-first-token latency in
two actual six-profile native trials. It does not deliver a decode speedup
or establish a new vLLM ratio. Text B1 TTFT regresses in both trials; this
candidate remains OFF by default rather than replacing the general bundle.
The overall 2x-decode/half-TTFT goal remains unmet.

Control and candidate both enable screened greedy heads, ordered prefill
streaming and partial cache reset. Only the candidate enables serial vision
prefetch. Original learned weights, FP32 contraction order, image patches,
prompts, token caps, stop semantics and complete generated outputs remain
unchanged. The separate shader library disables fast math. Admission is
limited to 216 vision tokens, hidden width 768, 12 heads and batch 1 or 2;
other shapes use the original route.

Validation PID 64980 exited 0 in 1309.0726 s, passing all 27 stages: the
original 19, the actual trained full-grid gate, 51 trained screening queries,
and six recovery stage suites. The full-grid gate checks B1/B2, small-grid
fallback, two fresh encodes per policy and cleared counters after close.
All 55,296 / 110,592 full-grid embedding values are bit-identical to control;
the small grids also match exactly. Candidate dispatch counts are 12 for the
full grids and zero for the small grids. Every native image request records
12 actual dispatches; the control records zero.

Initial native PID 69320 exited 0 in 1241.9312 s including the dependency
wait. Confirmation PID 91590 exited 0 in 159.6185 s. Independent audits check
1,812 archived files, all original gates, both recovery suites and 144 complete
matching outputs across the two trials. Each trial runs fresh sequential
control/candidate/candidate/control workers, three requests per worker. The
first request is reported separately; four warm requests per policy/profile
form each mean. Inference runtimes never intentionally overlap; guards sample
every 0.5 s. Every image request performs fresh encoding with the same combined
threaded/striped preprocessing in both policies.

| Model / batch | Initial TTFT control / candidate ms | Repeat TTFT control / candidate ms | Repeat decode control / candidate tokens/s per row |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B / 1 | 35.651 / 36.292 | 35.537 / 36.966 | 64.184 / 64.848 |
| Qwen3-0.6B / 2 | 57.680 / 56.881 | 56.525 / 56.727 | 61.448 / 60.606 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 28.137 / 27.891 | 27.597 / 27.206 | 78.800 / 78.877 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 45.089 / 44.382 | 44.575 / 45.359 | 76.260 / 76.451 |
| Qwen3.5-0.8B VL / 1 | 175.225 / 168.629 | 174.381 / 168.803 | 52.501 / 52.833 |
| Qwen3.5-0.8B VL / 2 | 332.695 / 322.129 | 334.298 / 324.752 | 48.288 / 48.460 |

VL TTFT falls 3.76% / 3.20% at B1 and 3.18% / 2.86% at B2 in the initial /
repeat trials. Repeat vision-stage wall time including reset falls from
59.946 to 55.034 ms and 125.762 to 116.165 ms, respectively. Text B1 TTFT
rises 1.80% / 4.02%; its decode direction changes between trials. Text B2
decode also changes direction. Other text/coder changes are small and mixed.
These observations do not establish the cause of non-vision variations.

Separate clocks, initialization and first request after initialization,
partial reset bytes, prefill submission counts and peak individual child RSS
are in `native-phases.json` and `confirmation-native-phases.json`. Clocks may
overlap and must not be summed. RSS is not unique Metal memory. Caches were
not flushed, so first use after initialization is not a hardware-cold claim.
These native-only results must not be combined with older vLLM measurements
to fabricate a fresh comparator ratio.

Candidate library SHA-256:
`b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d`.
Control library SHA-256:
`75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e`.
Initial archive SHA-256:
`1fff95e7af82d55a8331c6b7cb0455dfab828d6791237fc3c1e43fc3c7666dda`.
Combined archive SHA-256:
`9776c61a8b850264c6f1eab988db124ba2cf3a8c652e77ee7001dc32a2936e48`.

```text
python3 launch_serial_vision_checks.py ROOT ROOT/logs/serial-vision-checks-20261015 checks
python3 queue_serial_vision_profiles.py ROOT ROOT/logs/serial-vision-native-20261015
python3 archive_serial_vision.py ROOT ARCHIVE
python -X utf8 verify_serial_vision.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_serial_vision.py ROOT ROOT/logs/serial-vision-native-confirm-20261015
python3 archive_serial_vision.py ROOT ARCHIVE serial-vision-native-20261015 serial-vision-native-confirm-20261015
python -X utf8 verify_serial_vision.py EVIDENCE CONFIRMATION_JSON serial-vision-native-20261015 serial-vision-native-confirm-20261015
```
