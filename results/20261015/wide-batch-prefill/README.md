# Exact 160-row VL batch-2 prefill investigation

The full 2x decode / half-TTFT goal remains unmet. The first audited native
trial reduced VL B2 warm TTFT by 8.25%, from 324.36 to 297.60 ms, with decode
50.23 to 49.75 tokens/s per row. Confirmation reduced TTFT by 7.14%, from
321.21 to 298.26 ms, with decode50.07 to49.61 tokens/s per row. The option remains
OFF by default; results are from the same M4 16 GiB Mac mini.

The existing decoder projection limit of 128 physical rows prevents two
80-token prompt chunks from sharing one prefill pass. The candidate admits
exactly 160 rows in addition to the existing range. It keeps the unchanged
80-token logical chunk limit, original checkpoint values, projection kernels,
FP32 arithmetic and original numerical bounds. Batch-2 chunk80 requires the
explicit wide-batch build flag, resident shared weights, chunk recurrence and
hidden width at most 1024. Other unsupported profiles remain rejected.

Actual Mac CTest execution passed all five native tests, including the new
30-case wide projection test: exact-half MPS and raw BF16 fallback, FP64
reference bounds, pending-output reads, abort/recovery, uninitialized inputs
and rejection of rows129/159/161. This is a primitive result, not a complete
model or latency result.

Validation PID 69725 exited 0 in 1412.9496 s: the original 27 stages followed by four
first/last trained linear/full-attention references at batch2 chunk80, with
three carried steps each and the unchanged `5e-6 + 5e-5 * abs(expected)` bound.
The worst new trained-reference bound ratio was 0.5432836183. The full native
profile process PID72436 subsequently exited 0 after its dependency wait and
all six profiles (1443.4071 s including wait). An independent audit verified
2699 archived files, 72 complete matching outputs, four new reference profiles
and 12 carried steps, existing full-grid/head/recovery gates, actual route and
clear counts, and a 982-file source closure.

| Model | Batch | Control / candidate tokens/s per row | Control / candidate warm TTFT ms |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 63.7358 / 62.7519 | 36.7575 / 36.6458 |
| Qwen3-0.6B | 2 | 61.1090 / 60.6480 | 59.7118 / 60.2529 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 79.2440 / 80.5196 | 26.6692 / 26.1835 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 76.8213 / 77.2676 | 41.5458 / 40.8016 |
| Qwen3.5-0.8B VL | 1 | 53.9176 / 54.5199 | 178.2180 / 186.2046 |
| Qwen3.5-0.8B VL | 2 | 50.2324 / 49.7466 | 324.3634 / 297.5964 |

Other profiles are mixed: text decode is lower and VL B1 TTFT is higher in
this trial. This is not a universal improvement. Initialization, first request
after initialization, RSS and all raw means are preserved in verification.json
and the archived worker logs; no hardware-cold total is inferred.

Initial archive SHA256:
`1b01d1820fd55313996c3044933ca7484da90ea1f4f18f877465b89410531c40`.
Full six-profile confirmation PID98672 exited0 in166.8254s. The combined
archive contains3015 verified files and144 complete matching requests across
both native trials. Model/checkpoint/config/tokenizer identities match across
the two trials. The four new references and12 carried steps are reused rather
than counted twice. Actual candidate VL B2 wide dispatches were186 per request,
with5 partial submissions, in both trials.

| Model | Batch | Repeat control / candidate tokens/s per row | Repeat control / candidate TTFT ms |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 62.6562 /63.5649 |36.3037 /36.4154 |
| Qwen3-0.6B | 2 | 61.0681 /60.1901 |60.2875 /59.7528 |
| Qwen2.5-Coder-0.5B-Instruct |1 |80.3401 /80.5875 |26.0888 /26.5911 |
| Qwen2.5-Coder-0.5B-Instruct |2 |75.7743 /77.2865 |43.4061 /42.5261 |
| Qwen3.5-0.8B VL |1 |54.4973 /54.8399 |184.6235 /183.1303 |
| Qwen3.5-0.8B VL |2 |50.0743 /49.6081 |321.2059 /298.2614 |

Combined archive SHA256:
`f5227219ccc33bee44b9ab0c99abe26ddd0b3e11d4ec2491d5c0ea143f8bcac7`.
The independent initial/repeat audits and combined-verification.json preserve
all results. These are native comparisons; a fresh actual vLLM-Metal study is
running as PID3095 in `logs/wide-batch-vllm-20261015`. Keep its source and
drivers unchanged until terminal and archived.

The native ABBA study preserves all six external model/batch workloads
and their complete expected outputs. Only candidate VL B2 changes its internal
plan from `[32,32,16]` to `[80]`; total physical prompt slots remain80.
Control uses the archived interleaved-index library
`837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471`.
Actual wide projection dispatch counts are published after submission and
cleared on begin/reset/abort. Both runtimes run sequentially. Initialization,
first request after initialization and warm requests remain separate; caches
are unflushed. This study does not establish a new vLLM comparison ratio.

Replaced source bytes are preserved remotely in
`logs/wide-batch-prefill-original-source-20261015`. The prior native and
prefill-replan jobs were terminal and independently archived before install.
The original Mac checkout, GPT-OSS and Pi checkouts were not changed.

```text
python -X utf8 test2026/metal/prepare_wide_batch_prefill_sources.py
python3 install_wide_batch_prefill_sources.py ROOT
python3 launch_wide_batch_prefill_checks.py ROOT ROOT/logs/wide-batch-prefill-checks-20261015 checks
python3 queue_wide_batch_prefill_profiles.py ROOT ROOT/logs/wide-batch-prefill-native-20261015
# Only after both jobs have terminal successful status:
python3 archive_wide_batch_prefill.py ROOT ARCHIVE
python -X utf8 verify_wide_batch_prefill.py EXTRACTED_EVIDENCE VERIFICATION_JSON
```
