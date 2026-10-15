# Native lossless BF16 blocks: repeated full-model decode gains

Validation PID88930 exited0 in1419.8316s. Benchmark PID88983 exited0 in1591.2531s including dependency wait. Independent verification checks2845 archived files, all28 validation stages,51 trained-head queries, trained vision-grid and recovery cases, four wide trained references with12 carried steps, and72 complete matching generated outputs. Actual coalesced decode counters are positive and stable only for the candidate; first-prefill counters remain zero. All seven CTests passed, including80 new primitive cases. Original numerical bounds are unchanged.

The candidate combines the prior four-output interleaved route with exact coalesced BF16 block storage at a fixed64-thread policy. Its control is the retained wide-prefill library, not the interleaved-only candidate. Thus this table does not isolate the incremental block gain from the prior interleaved gain. Original BF16 values are reconstructed before FP32 arithmetic. Extra raw/interleaved/compact buffers remain resident.

| Model / batch | Decode tokens/s control -> candidate | TTFT ms control -> candidate | Peak individual RSS GiB control -> candidate | Init s control -> candidate |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B / 1 | 63.6223 -> 73.8948 | 37.4769 -> 36.1065 | 2.280 -> 3.654 | 2.329 -> 3.512 |
| Qwen3-0.6B / 2 | 61.2447 -> 70.3273 | 58.7154 -> 56.9194 | 2.373 -> 3.725 | 2.358 -> 3.533 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 79.0375 -> 87.4888 | 26.5685 -> 27.9146 | 1.931 -> 2.972 | 1.976 -> 2.812 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 76.8016 -> 83.7684 | 42.2032 -> 43.7561 | 1.931 -> 2.972 | 1.966 -> 2.812 |
| Qwen3.5-0.8B / 1 | 53.2812 -> 60.3025 | 173.8367 -> 180.6539 | 3.363 -> 5.003 | 4.189 -> 5.547 |
| Qwen3.5-0.8B / 2 | 50.6380 -> 55.2689 | 301.6679 -> 296.0239 | 3.404 -> 5.054 | 4.216 -> 5.491 |

Decode gains range9.1–16.1% in this first native cohort. TTFT changes are mixed, including coder and VL B1 regressions. Added peak individual process RSS is about1.04–1.65GiB; initialization adds about0.84–1.36s. These are process measurements, not unique Metal allocations. Initialization and first-after-init requests remain separate from warm means. OS/driver caches are not flushed; no hardware-cold result is claimed.

Initial archive SHA256: `3641ad8f1324f6033f69db7aa1940b2a0eb8ba797bf2e8745c404811ed4846fe`.
Control library SHA256: `9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b`.
Candidate library SHA256: `c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1`.

CPU full-checkpoint coverage:378 admitted matrices (168text/72coder/138VL), with exact byte-plane reconstruction. Compact block fractions96.44%/95.99%/96.31%. One-copy extra buffer estimates591.06/451.21/705.66MiB are distinct from actual process RSS. `checkpoint-coverage/` preserves the executed source and report.

Confirmation PID19668 exited0 in178.8188s. Independent audits check3159 combined archived files,72 initial outputs and72 confirmation outputs. The identity join verifies identical source/library/asset/case records,144 complete matching outputs and repeated decode gains for all six profiles. The original numerical gates are reused, not counted as independent repeated validation.

| Model / batch | Repeat decode tokens/s control -> candidate | Repeat TTFT ms control -> candidate |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 63.2732 -> 73.9942 | 38.1031 -> 35.6289 |
| Qwen3-0.6B / 2 | 60.4640 -> 71.3397 | 59.2019 -> 56.2920 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 78.8289 -> 88.1912 | 26.0745 -> 27.0199 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 76.9586 -> 84.5282 | 41.3720 -> 42.1309 |
| Qwen3.5-0.8B / 1 | 53.8944 -> 61.6159 | 173.3776 -> 183.1336 |
| Qwen3.5-0.8B / 2 | 50.4445 -> 55.2427 | 300.4442 -> 295.3271 |

Across the two trials, text decode gains16.1–16.9% B1 and14.8–18.0% B2; coder gains10.7–11.9% B1 and9.1–9.8% B2; VL gains13.2–14.3% B1 and9.1–9.5% B2. Text TTFT and VL B2 TTFT improve in both trials; coder and VL B1 TTFT regress in both. Extra initialization and process memory remain real tradeoffs. No broad default promotion follows.

Combined archive SHA256: `6a63f6aa003faed06fff7fd318e987a5a9473d2eeaf8585298e74eb411c5be70`.
A fresh matched comparator cohort launched as PID23200 under `logs/coalesced-body-vllm-20261015`. Its results are pending; do not combine these native-only means with older vLLM means. Preserve the selected source/library records until that cohort is terminal and archived.

Full workloads remain unchanged, including both VL B2 chunk plans `[80]`; no prefix/vision reuse or speculation is enabled. Original Mac, GPT-OSS and Pi checkouts are preserved. No new vLLM ratio is inferred; the overall2x decode / half-TTFT goal remains unmet.

```text
python3 launch_coalesced_body_native_checks.py ROOT ROOT/logs/coalesced-body-native-checks-20261015 checks
python3 queue_coalesced_body_native_profiles.py ROOT ROOT/logs/coalesced-body-native-20261015
python3 archive_coalesced_body_native.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_coalesced_body_native.py RESULTS RESULTS/verification.json
python3 launch_confirm_coalesced_body_native.py ROOT ROOT/logs/coalesced-body-native-confirm-20261015 checks
```
