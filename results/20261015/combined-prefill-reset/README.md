# Combined bounded reset and ordered prefill: initial latency tradeoff

This experiment enables the two already implemented optional features in one
fresh build: `GARNET_METAL_DIRTY_PREFIX_RESET` and
`GARNET_METAL_PREFILL_STREAM`, with the selected global-half-alpha policy.
The initial complete comparison reduces warm first-token latency in all six
profiles, with a decode penalty. Keep both features optional and default OFF.
Independent predecessor gains are not added to these actual measurements.
There is no fresh vLLM comparison.

The previous streaming-only experiment passed all nineteen original gates,
both CTests and two complete six-profile runs (144 requests). It repeatedly
improved warm TTFT by 3–12%, but its repeat decode rates were 0.1–1.5% lower.
The separate bounded-reset experiment repeatedly reduced text reset latency,
with a small text B2 decode penalty. This combination investigates whether
both latency reductions survive together without changing model math.

Owned PID **27124** started at
`ROOT/logs/combined-prefill-reset-20261015` and exited 0 after 1370.33 seconds.
Full validation exited 0 after 1225.30 seconds, followed by about 145 seconds
of serving measurements. All original nineteen stages, three CTests and three
trained session recovery suites passed. Independent `verification.json`
verified 782 archive files, all twenty-two stages, six B1/B2 recovery results,
all 72 complete serving outputs and both features' actual counters.
The workflow creates `build-combined-prefill-reset` and preserves every older
build. Sources are pinned to the completed streaming manifest; it changes
build options and investigation helpers only.

The durable launcher runs the complete original nineteen validation stages,
three CTests (numerical reference, reset, streaming) and all three trained
session abort/recovery suites. Only after those finish with exit 0 does it
run all six original ABBA serving profiles, with four warm requests per policy
and profile across two fresh workers. No comparator or other GPU probe is
started alongside validation or serving. Guards sample every 0.5 seconds,
reject competing inference/probe jobs and only stop owned processes.

Serving asserts full generated tokens, prompt IDs, finished flags, fresh image
hashes, unchanged workload caps and chunk plans, real partial submission
counts, zero decode partial submissions, and the original exact reset extent,
state, fallback and byte-count formulas. Unrelated optional features remain
off. The initial result is independently archived and audited. A fresh repeat
remains required before selection; this does not establish fresh vLLM ratios
or the full 2x decode / half-TTFT goal.

| Model / batch | Control TTFT ms | Candidate TTFT ms | Control tokens/s per row | Candidate tokens/s per row | Decode candidate/control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text / 1 | 50.706 | 39.878 | 59.042 | 57.472 | 0.9734 |
| Text / 2 | 83.810 | 60.453 | 57.817 | 55.701 | 0.9634 |
| Coder / 1 | 33.375 | 28.012 | 72.147 | 71.572 | 0.9920 |
| Coder / 2 | 52.656 | 45.109 | 70.468 | 70.258 | 0.9970 |
| VL / 1 | 198.726 | 189.438 | 48.102 | 47.862 | 0.9950 |
| VL / 2 | 361.248 | 348.484 | 44.394 | 44.340 | 0.9988 |

Warm TTFT reductions are 21.4%, 27.9%, 16.1%, 14.3%, 4.7% and 3.5%.
Decode is 2.7%, 3.7%, 0.8%, 0.3%, 0.5% and 0.1% slower respectively.
These are four warm requests per policy/profile from two fresh workers,
excluding each first request; batch-2 aggregate throughput is twice the row
rate. Text B2 loses more decode speed than either predecessor separately;
the combination is not selected as a throughput gain.

`phases.json` shows text B1 reset 7.490→1.041 ms and prefill 41.456→37.306 ms;
text B2 reset 23.906→1.609 ms and prefill 58.415→57.132 ms. Text B2 GPU
prefill span rises 50.960→55.159 ms despite lower end-to-end TTFT. Clocks
overlap and must not be added or attributed to isolated kernels. First
request AFTER initialization improves in all six profiles; initialization is
recorded separately and OS/driver caches were not flushed.

Initial archive SHA-256:
`6edb521245d913c8dd7af1b9ea4d5fa1bfcacc0c202ce78c50d9ce4097ffcb77`.
Candidate library/source identities are pinned by archived checks metadata.
Partial counts are 6 for text, 5 for coder/VL B1 and 15 for VL B2; control
and decode counts are zero. Reset byte/state/fallback counts passed the exact
independent formula audit on every request.

Expanded wrapper syntax and both feature admission formulas were checked
locally before launch. Exact outer helpers and source identities are saved
by the launcher; the guard saves executed model validation source, stage
commands, terminal status, monitor samples and library/source hashes.

```text
python3 launch_combined_prefill_reset.py ROOT ROOT/logs/combined-prefill-reset-20261015
python3 archive_combined_prefill_reset.py ROOT ARCHIVE
python verify_combined_prefill_reset.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_combined_prefill_reset.py ROOT OUTPUT
python verify_combined_prefill_reset.py EVIDENCE CONFIRMATION_JSON --confirm
```

The original GPT-OSS and Pi checkouts were inspected and remain clean on
`feature/gpt-oss-120b` and `feature/qwen35-vl-pi5b` respectively. This work
is isolated in `feature/qwen35-metal`. The full optimization goal remains
active and unmet.

The independent audit retains all twenty-two validation stages, both feature
CTest signals, all six B1/B2 recovery results, the original full serving
outputs and frame hashes, both features' actual counters and independently
recomputed means. Archive and expanded-verifier syntax were checked before
use; no pending job is represented as verified terminal evidence.

Owned queue **33256** waited for this exact dependency, archived its terminal
successful evidence, then ran the separate eight-lane component probe. It
exited 0 after 1082.43 seconds including the wait. All 136 component checks
passed, but the kernel was slower and was rejected without integration.
No probe overlapped combined validation or serving.

Fresh full-six repeat PID **52435** started only after the independent initial
audit was uploaded and queue **33256** reached terminal status.
`ROOT/logs/combined-prefill-reset-confirm-20261015` uses fresh sequential
workers and the unchanged candidate/source identities. Terminal repeat timing
and independent audit are pending; do not represent them as measured success.
