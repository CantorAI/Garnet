# Combined bounded reset and ordered prefill: full validation pending

This experiment enables the two already implemented optional features in one
fresh build: `GARNET_METAL_DIRTY_PREFIX_RESET` and
`GARNET_METAL_PREFILL_STREAM`, with the selected global-half-alpha policy.
There is no new combined performance result yet. Independent gains must not
be added or presented as a measured combined result.

The previous streaming-only experiment passed all nineteen original gates,
both CTests and two complete six-profile runs (144 requests). It repeatedly
improved warm TTFT by 3–12%, but its repeat decode rates were 0.1–1.5% lower.
The separate bounded-reset experiment repeatedly reduced text reset latency,
with a small text B2 decode penalty. This combination investigates whether
both latency reductions survive together without changing model math.

Owned PID **27124** started at
`ROOT/logs/combined-prefill-reset-20261015`. A real `ps` observation confirmed
the handle live; configure exited 0 and the isolated build was running.
This is historical launch evidence, not proof of current liveness or success.
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
off. A successful run still requires independently archived and audited
evidence and a repeat before selection; this does not establish fresh vLLM
ratios or the full 2x decode / half-TTFT goal.

Expanded wrapper syntax and both feature admission formulas were checked
locally before launch. Exact outer helpers and source identities are saved
by the launcher; the guard saves executed model validation source, stage
commands, terminal status, monitor samples and library/source hashes.

```text
python3 launch_combined_prefill_reset.py ROOT ROOT/logs/combined-prefill-reset-20261015
python3 archive_combined_prefill_reset.py ROOT ARCHIVE
python verify_combined_prefill_reset.py EVIDENCE VERIFICATION_JSON
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

Owned queue **33256** is waiting for this exact live dependency. It archives
successful terminal combined evidence before starting the separate eight-lane
decode component probe. It never starts a probe alongside this job and does
not interrupt the dependency. A dependency failure stops the queue.
