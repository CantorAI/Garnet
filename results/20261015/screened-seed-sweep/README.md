# Exact screened-head seed-width exploration

Actual PID 94813 exited 0 in 147.8686 s. Four sequential component workers
test exact seed widths 8192, 1024, 4096 and 32768 against the original full
BF16/FP32 head. All 204 captured-query checks, 312 row states and 132 unsafe
input/fallback checks pass. Every retained score is bit-identical to full
projection; every sampled rejected score is below the actual seed threshold;
greedy predictions match. The existing strict-math bounds and original
refinement are unchanged. This does not certify bounds for all future inputs.

Each width uses the same byte-verified immutable trained head/query assets.
The derived INT8 index only filters rows; it is not a replacement model.
The audit reconstructs exact shader/harness bytes, checks 143 archived files,
source and asset hashes, guard traces, ABBA ordering and 13,056 raw timed
component executions, with eight samples per window and two ABBA rounds.

The table averages the eight ordinary captured steps per model/batch,
excluding the separate mixed-row check. Each fraction is screened component
wall time / original full-head wall time within that width's worker.

| Model / batch | Seed 8192 | Seed 1024 | Seed 4096 | Seed 32768 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B / 1 | 0.78679 | 0.76195 | 0.77231 | 0.85061 |
| Qwen3-0.6B / 2 | 0.81385 | 0.79931 | 0.81030 | 0.90386 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 0.79522 | 0.76278 | 0.78182 | 0.85458 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 0.81470 | 0.79725 | 0.80507 | 0.89183 |
| Qwen3.5-0.8B / 1 | 0.80857 | 0.80307 | 0.80470 | 0.85075 |
| Qwen3.5-0.8B / 2 | 0.80330 | 0.78402 | 0.79526 | 0.85365 |

The 1024-row seed has the lowest normalized GPU and wall fraction in all six
groups in this first study. It retains roughly 1026–1036 rows per ordinary
query instead of 8192–8194; refinement of the few extra candidates preserves
the sampled answers. The larger seed is slower and is rejected. Separate
workers and raw-head normalization are not a direct same-process 8192-versus-
1024 benchmark. That direct ABBA comparison is the next gate before native
integration. No new whole-model TPS, TTFT or vLLM ratio follows from this study.
The subsequent [same-process ABBA and repeat](../screened-seed-abba/README.md)
confirm a small component reduction. A separate OFF-by-default native option
is now under full-model validation; its serving impact remains unmeasured.

Archive SHA-256:
`caf5353e24cca3bb133efd504e4788bd43f91ad844807859170c77c6b3a4ba2c`.

```text
python3 launch_screened_seed_sweep.py ROOT ROOT/logs/screened-seed-sweep-20261015 checks
python3 archive_screened_seed_sweep.py ROOT ARCHIVE
python -X utf8 verify_screened_seed_sweep.py RESULTS VERIFICATION_JSON
```
