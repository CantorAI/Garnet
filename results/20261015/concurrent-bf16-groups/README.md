# Independent native projections in one compute pass: mixed results

The actual two sequential component trials passed every complete output-bit,
padding, fresh-input and original FP64 check. The independent audit verified
48 archived files, 32 group/batch configurations across the two trials, 30,595
validated executions and 6,144 timed executions. A concurrent pass does not
produce a broad repeated gain, so this is not enabled in serving. No new
whole-model or vLLM speed ratio is claimed. The full goal remains unmet.

The preceding native inference telemetry places text/coder warm decode host
preparation around 0.13–0.16 ms and encoding around 0.48–0.55 ms, compared with
12.6–16.0 ms observed GPU spans. These clocks overlap and must not be added.
This motivates testing GPU scheduling before spending effort on host-list or
control-upload changes. The source data is the control records in
`../reused-zero-source-inference/evidence/logs/reused-zero-source-confirm-native-20261015/jobs`.

This component preserves the exact current native BF16 decode shaders and
default compiler options. It changes only compute-pass scheduling for independent
learned gate/up pairs and dense Q/K/V groups, plus an irregular synthetic
read-only-input pair. VL layer 0 already has a combined QKV projection, so it
has no separate Q/K/V group in this test. B1 and B2 use fresh original, negated,
half and doubled FP32 inputs with negative-zero elements. All outputs are
separate allocations; there are no dependencies among dispatches in a group.

The control uses one serial encoder per projection. Mode 1 uses one serial
encoder for the group, isolating encoder reuse. Mode 2 requests one concurrent
encoder for the group. The API allows concurrent compute commands; requesting
it does not prove hardware overlap. Dependencies on later consumers would still
need correct synchronization in any future integration. See Apple's
[resource synchronization documentation](https://developer.apple.com/documentation/metal/resource-synchronization?language=objc).

Candidate/control component wall-time fractions below 1 are faster. These
are full grouped-projection wall intervals including encode/submit/wait,
not model TPS or TTFT. All raw per-window GPU and encoding times are preserved.

| Model | Batch | Group | Concurrent wall fraction, initial / repeat |
| --- | ---: | --- | ---: |
| Qwen3-0.6B | 1 | gate/up | 0.89129 / 0.98341 |
| Qwen3-0.6B | 2 | gate/up | 1.01223 / 1.08145 |
| Qwen3-0.6B | 1 | Q/K/V | 1.05107 / 0.86586 |
| Qwen3-0.6B | 2 | Q/K/V | 1.03185 / 1.00311 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | gate/up | 1.00727 / 1.02652 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | gate/up | 0.85717 / 1.02814 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | Q/K/V | 0.95561 / 1.01405 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | Q/K/V | 1.07925 / 1.02359 |
| Qwen3.5-0.8B | 1 | gate/up | 0.97441 / 0.99776 |
| Qwen3.5-0.8B | 2 | gate/up | 0.95535 / 0.92267 |

Text B1 and VL gate/up times are lower in both runs, with a nearly flat VL B1
repeat. VL B2 shows the clearest repeated component direction, 4.5% and 7.7%
lower wall time. Text B2 gate/up and multiple Q/K/V groups are slower; coder
B2's apparent initial gain reverses. Mode 1 serial reuse is also mixed in the
raw audit. These results do not justify a general scheduling change or show
that the remaining B1 decode / TTFT targets have been reached.

Both trials use at least 250 ms common checked warmup, two ABBA rounds per
mode, and 12 samples per window. Reference outputs use the unchanged
gamma(ceil(K/32)+32)*sumabs+1e-7 FP64 envelope, maximum ratio 0.0090262.
Every candidate warm/timed execution matches all reference output and padding
bits; no comparison tolerance or precision is reduced.

Actual job records:

| Trial | PID | Exit | Wall seconds |
| --- | ---: | ---: | ---: |
| concurrent-bf16-groups-20261015 | 85269 | 0 | 13.8496 |
| concurrent-bf16-groups-confirm-20261015 | 85681 | 0 | 13.1379 |

The first trial was independently audited before the repeat. Only one owned
GPU probe runs at a time; 0.5-second inference/probe guards retain all original
exclusions and reject unknown workers. No comparator runs alongside either
trial. GPT-OSS and Pi checkouts are unchanged; production selection is unchanged.

Initial archive SHA-256:
`a14d6d48de6a446b1734976fca73b16fddab73e7521380fc947e2f121661fdb3`.
Combined archive SHA-256:
`dc6c4a9cfa5b7459e143a622d5fa2f79f2228681250613db6e4f5c297cad0d6a`.
Exact executed source, raw timing arrays, commands, versions, checkpoint and
tensor identities and process monitors are in `evidence/`; `verification.json`
recomputes every reported mean and fraction. Weight arrays and binary are
omitted, with their hashes retained.

```text
python3 launch_concurrent_bf16_groups.py ROOT ROOT/logs/concurrent-bf16-groups-20261015
python3 archive_concurrent_bf16_groups.py ROOT ARCHIVE
python -X utf8 verify_concurrent_bf16_groups.py RESULTS VERIFICATION_JSON
```
