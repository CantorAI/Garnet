# Fresh Garnet / vLLM-Metal comparison after private KV writes

The retained native library is compared with actual vLLM-Metal on the same
Apple M4 Mac mini, 16 GiB, macOS 15.6.1. This is an additional matched-FP32
study: native original BF16 storage with FP32 operations versus losslessly
expanded, bit-equivalent F32 learned values with FP32 comparator operations.
It does not establish stock/default BF16 speedups. All original source and
expanded checkpoint hashes and complete learned-value proofs are retained.

Fresh actual loaded-model audits verify 310 text, 290 coder and 473 VL MLX
parameter tensors are F32. Probed activations and runner KV caches are also
F32 for all three models. Configuration settings alone are not used as proof.
Comparator versions remain vLLM 0.30.0+cpu, vLLM-Metal 0.30.0, MLX 0.32.1,
MLX-LM 0.32.0, MLX-VLM 0.6.17, Torch 2.13.0 and Transformers 5.19.0.
The CPU Torch wheel hosts an actual MLX GPU execution path; the version label
is not evidence that these model computations run on the CPU.

## Actual warm results

| Model / batch | Garnet steps/s/row | vLLM steps/s/row | Decode ratio | Garnet TTFT ms | vLLM TTFT ms | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 text / 1 | 59.78 | 37.81 | 1.581 | 51.88 | 75.07 | 0.691 |
| Qwen3 text / 2 | 58.36 | 25.96 | 2.248 | 91.29 | 97.58 | 0.936 |
| Qwen2.5 coder / 1 | 72.10 | 47.37 | 1.522 | 39.73 | 58.58 | 0.678 |
| Qwen2.5 coder / 2 | 71.61 | 28.64 | 2.500 | 64.65 | 76.36 | 0.847 |
| Qwen3.5 VL / 1 | 48.15 | 26.60 | 1.810 | 220.21 | 266.43 | 0.827 |
| Qwen3.5 VL / 2 | 44.31 | 17.30 | 2.561 | 391.57 | 432.93 | 0.904 |

Batch-2 aggregate output throughput is twice the per-row step rate. All
batch-2 decode ratios exceed 2 in this precision study. All batch-1 ratios
remain below 2, and no first-token fraction is at most 0.5. The full goal
therefore remains active and unmet; these are not default BF16 conclusions.

Each profile executes sequential Garnet / vLLM / vLLM / Garnet process order,
with two fresh persistent processes per runtime, three requests per process,
and four non-first requests per warm mean. All 72 selected requests preserve
complete normalized token IDs, prompt IDs, finished flags and one prediction
per step. The comparator's omitted terminal stop ID is restored only using
its recorded stop reason or verified default EOS. Text/coder retain the same
22/26-token prompts and cap 64. VL retains the same fresh 288x192 RGB/FP32
patch bytes, 54 visual tokens, 79-token prompt and cap 32. Native VL uses
chunk 80 at batch 1 and chunk 32 at batch 2. Every frame is freshly encoded;
comparator encoder identities are unique across rows/requests. Prefix caching
and speculative decoding are disabled and verified in actual engine records.

## Initialization and memory

| Model / batch | Garnet init s | vLLM init s | Garnet peak child RSS GiB | vLLM peak child RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 2.53 | 18.03 | 2.276 | 3.350 |
| Text / 2 | 1.77 | 16.99 | 2.320 | 3.497 |
| Coder / 1 | 1.39 | 15.98 | 1.927 | 3.579 |
| Coder / 2 | 1.39 | 15.93 | 1.927 | 3.559 |
| VL / 1 | 4.09 | 33.73 | 3.361 | 5.521 |
| VL / 2 | 4.15 | 33.21 | 3.397 | 5.391 |

Initialization and first-use requests are saved separately from warm timing.
Filesystem/driver caches were not flushed, so fresh processes are not proof
of cold hardware. RSS is RUSAGE_CHILDREN's high-water individual child value,
not aggregate worker RSS or unique GPU memory. Peak RSS includes startup;
these figures are not pure inference allocations. Native and comparator
weight storage widths differ despite exact learned-value equivalence.

## Remaining budgets and execution guards

remaining-budgets.json derives each unmet deadline from its actual comparator
measurements and retains native phase counters. For text batch 1, the decode
target is 75.62 steps/s and TTFT target 37.53 ms. Native prefill GPU time is
37.35 ms before the other work, while CPU encoding is 5.20 ms. Decode GPU
time is 15.53 ms, with CPU preparation/encoding about 0.61 ms combined.
GPU execution remains material; removing CPU work alone would not satisfy
the requested decode target. Wait time overlaps GPU time and is not additive.

Before/after guards and 0.5-second monitors reject other inference runtimes.
Only the process group created for the current worker is exempt; unrelated
native and Python comparator processes remain visible. Four ownership tests
verify this filtering, including rejection of multiple owned native workers.
The selected runtimes never run together. One coder batch-1 attempt was
interrupted by an unknown external xlang3 process and excluded; its failure
and partial records remain. A five-second clear guard interval precedes the
retry. No external job was stopped. Sampling cannot prove the absence of
arbitrarily brief external jobs between polls.

completed/ retains exact commands/exits, monitors, guards, runtime versions,
actual dtype reports, full request records, checkpoint equivalence proofs and
warm/first-use measurements. source/ retains executed runner/worker/auditor
versions. Raw image tensors remain on the Mac with sizes/hashes in
evidence-proof.json. The native library remains
73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2.
Transferred archive SHA256:
a36ada17e1ccd2511a84b63ab3020815e936e3fcfcf616064e0a3c33fa515c1c.

Neither the primary GPT-OSS checkout nor the Pi checkout was modified.
