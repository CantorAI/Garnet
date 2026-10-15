# Exact 160-row VL batch-2 prefill investigation

The full 2x decode / half-TTFT goal remains unmet. This candidate has no
measured serving speed claim yet. It is OFF by default and is being validated
on the same M4 16 GiB Mac mini.

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

Validation PID 69725 is running the original 27 stages followed by four
first/last trained linear/full-attention references at batch2 chunk80, with
three carried steps each and the unchanged `5e-6 + 5e-5 * abs(expected)` bound.
The queued profile process PID 72436 requires that exact validation process
to terminate successfully with all 28 stages before starting inference.
Do not edit its pinned source or drivers while either process is live.

The queued native ABBA study preserves all six external model/batch workloads
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
