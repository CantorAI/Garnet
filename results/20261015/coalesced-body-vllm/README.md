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

Confirmation PID34285 exited0 in612.8577s. Independent audit verifies1493 confirmation files,72 matching outputs and three fresh GPU dtype audits. The join verifies identical selection, versions, sources, assets and case files across144 complete matching outputs and six GPU dtype audits. In both cohorts,4/6 profiles meet2x decode,2/6 meet half-TTFT and0/6 meet both. Text B1 falls from2.0392x initially to1.9848x on repeat; it is not rounded into a pass.

| Model / batch | Repeat Garnet / vLLM row tokens/s | Repeat Garnet / vLLM TTFT ms | Repeat decode ratio | Repeat TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B / 1 | 74.5058 / 37.5380 | 35.4438 / 78.2122 | 1.9848 | 0.4532 |
| Qwen3-0.6B / 2 | 69.9298 / 25.4279 | 57.4169 / 97.6511 | 2.7501 | 0.5880 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 87.2412 / 46.9444 | 27.5541 / 58.1109 | 1.8584 | 0.4742 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 83.7879 / 28.4397 | 43.4840 / 81.5716 | 2.9462 | 0.5331 |
| Qwen3.5-0.8B / 1 | 60.4676 / 26.2507 | 177.3791 / 254.3347 | 2.3035 | 0.6974 |
| Qwen3.5-0.8B / 2 | 55.1111 / 17.0851 | 299.1103 / 429.1466 | 3.2257 | 0.6970 |

The repeat guard rejected text B1 attempt0 on detecting inference-worker PID36306. Its timings are excluded; the clean attempt1 is used. The detected worker provenance is unproven. The complete rejected trace is retained in the confirmation archive, alongside all accepted guarded workers. No tolerance or guard was relaxed.

Confirmation archive SHA256: `1eaad9bce510ea94d0ca18d3e55e587243ed1d18399d6e5ab761dbeef906d347`.
`remaining-budgets.json` preserves both cohorts and each remaining gap. The overall goal remains active and unmet.

```text
python3 launch_coalesced_body_vllm.py ROOT ROOT/logs/coalesced-body-vllm-20261015
python3 archive_coalesced_body_vllm.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_coalesced_body_vllm.py RESULTS RESULTS/verification.json
python3 launch_confirm_coalesced_body_vllm.py ROOT ROOT/logs/coalesced-body-vllm-confirm-20261015
```
