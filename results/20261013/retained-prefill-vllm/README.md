# Fresh comparison with the retained dense prefill policies

All six profiles were executed again, with three fresh loaded-dtype audits.
The native library remains SHA256
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
Text uses 24 slots at batch 1 and opt-in 22 at batch 2; coder uses opt-in
26 at both batches. VL retains 80 / 32+32+16. The rejected text-head
prototype is absent. All checkpoints, prompts, image pixels, capacities,
output limits and stop rules are unchanged.

This is an additional matched-FP32 study on the same M4 / 16 GiB Mac mini.
Native original BF16 storage has exact learned values and FP32 arithmetic;
vLLM-Metal loads the losslessly expanded, bit-equivalent FP32 checkpoint.
Actual audits verify 310 text, 290 coder and 473 VL parameter tensors are
FP32, as are the probed activations and runner KV caches. Actual MLX GPU
execution is used. These results do not establish stock BF16 speedups.

| Model | Batch | Garnet steps/s/row | vLLM steps/s/row | Decode ratio | Garnet TTFT ms | vLLM TTFT ms | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 60.04 | 37.90 | 1.584 | 51.22 | 77.24 | 0.663 |
| Qwen3-0.6B | 2 | 58.38 | 25.50 | 2.289 | 85.82 | 97.65 | 0.879 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 71.19 | 47.15 | 1.510 | 40.37 | 56.12 | 0.719 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.52 | 28.38 | 2.485 | 62.06 | 83.09 | 0.747 |
| Qwen3.5-0.8B | 1 | 48.31 | 26.17 | 1.846 | 225.22 | 266.01 | 0.847 |
| Qwen3.5-0.8B | 2 | 44.82 | 17.17 | 2.610 | 388.41 | 443.05 | 0.877 |

All batch-2 decode ratios exceed 2; all batch-1 ratios remain below 2.
No TTFT fraction is at most 0.5. The full goal remains active and unmet.
Batch-2 aggregate throughput is twice the per-row rate. These fresh
comparator measurements update the earlier resident-KV comparison; small
differences between separate runs are not attributed to new algorithms.

Each profile uses sequential Garnet / vLLM / vLLM / Garnet order, two fresh
persistent workers per runtime and three requests per worker. Four non-first
requests form each warm mean. All 72 complete normalized output sequences,
prompt IDs, finished flags and prediction-step counts match the frozen
baseline. Missing comparator stop IDs are restored only from its recorded
integer stop reason or verified default EOS. Prefix reuse and speculation
are disabled. All VL frames are freshly prepared/encoded; comparator encoder
identities remain unique across rows and requests. No competing inference
was admitted. The three dtype audits and six comparisons have terminal exit 0.

Startup and first-use requests are separated in `first-use-summary.json`.
OS and driver caches were not flushed, so these fresh-model samples do not
establish cold hardware. Startup and first-request clocks are separate;
they are not silently combined into an observed end-to-end cold TTFT.
Individual child peak RSS is not aggregate or unique GPU memory.
`remaining-budgets.json` preserves current per-profile targets and measured
CPU/GPU phase counters. GPU execution overlaps submit/wait wall time.

## Commands and provenance

```text
python3 run_retained_prefill_vllm_profiles.py INVESTIGATION_ROOT OUTPUT_DIR
python3 compare_retained_prefill_vllm.py INVESTIGATION_ROOT OUTPUT_DIR MODEL BATCH
```

Exact commands, source/parameter/checkpoint/library hashes, learned-value
equivalence proofs, versions, actual dtypes, process guards, timing records
and complete outputs are under `evidence/`. Monitoring samples every 0.5 s;
arbitrarily short external jobs between polls cannot be ruled out. Only
owned worker groups are exempt, and only owned processes may be terminated.
The control library and other checkouts are preserved.

Archive SHA256:
`372fe9aabc8c61f90383343649c68075f7dcadbe09c605fc9717971b5bfd733d`.
