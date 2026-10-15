# Fresh lossless-body comparison against vLLM-Metal: initial cohort audited

Owned PID23200 exited0 in558.0408s. Independent audit verifies1457 archived files, three actual GPU dtype audits and72 complete matching outputs across six profiles. In this initial cohort,5/6 meet2x decode,2/6 meet half-TTFT, and1/6 meets both (text B1). Confirmation is required; the full goal remains unmet.

| Model / batch | Garnet / vLLM row tokens/s | Garnet / vLLM TTFT ms | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B / 1 | 74.9458 / 36.7522 | 35.6273 / 86.9596 | 2.0392 | 0.4097 |
| Qwen3-0.6B / 2 | 68.7498 / 23.6131 | 58.1542 / 102.3022 | 2.9115 | 0.5685 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 87.5102 / 46.9730 | 26.9278 / 57.8273 | 1.8630 | 0.4657 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 85.1040 / 28.3518 | 43.3857 / 86.5175 | 3.0017 | 0.5015 |
| Qwen3.5-0.8B / 1 | 61.2275 / 26.3286 | 171.4488 / 259.6041 | 2.3255 | 0.6604 |
| Qwen3.5-0.8B / 2 | 55.0251 / 17.0360 | 299.1174 / 428.5932 | 3.2299 | 0.6979 |

Decode is per batch row. Complete recorded normalized sequences match on both B2 rows, so aggregate throughput is twice the listed rate for these particular profiles. Do not generalize to unequal lengths or asynchronous arrivals. `initial-budgets.json` preserves exact means and gaps. Text/coder B2 and both VL profiles still miss half-TTFT; coder B1 still misses2x decode. A near-threshold value is not rounded into a pass.

Both runtimes use identical learned values and FP32 arithmetic. Comparator weights expand losslessly to FP32; Garnet retains exact BF16 representations and added buffers. This is not a stock BF16 vLLM or equal-storage comparison. Actual GPU audits confirm comparator parameter/activation/KV dtypes. Prompts, caps, stops and preprocessing match; images are freshly encoded. Prefix/vision reuse and speculation are disabled.

Each profile runs sequential G/V/V/G fresh workers, three requests per worker and four warm requests per policy. Initialization and first-after-init remain separate. OS/driver caches are not flushed. Peak process RSS/init costs are preserved in raw records and means; extra buffers are real costs. Guards exclude competing inference. No stale comparator timing is used.

Candidate library SHA256: `c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1`.
Initial archive SHA256: `0455d0a2c77c94609a2bab5c73396f2a0cc2ab76f6a6ad99819cb66b7fe5a95a`.

Confirmation launched as PID34285 at `ROOT/logs/coalesced-body-vllm-confirm-20261015` after independent initial audit and terminal diagnostic archival. Sources/library/selection remain frozen. The following hybrid graph study waits for exact PID34285 and a quiet interval. No repeated matched result is established yet.

```text
python3 launch_coalesced_body_vllm.py ROOT ROOT/logs/coalesced-body-vllm-20261015
python3 archive_coalesced_body_vllm.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_coalesced_body_vllm.py RESULTS RESULTS/verification.json
python3 launch_confirm_coalesced_body_vllm.py ROOT ROOT/logs/coalesced-body-vllm-confirm-20261015
```
