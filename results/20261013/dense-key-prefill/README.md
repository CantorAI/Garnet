# Dense one-pass prefill: verified improvement, full goal still unmet

Text and coder first-token latency improves when their 22/26-token prompts
use one padded 32-token pass instead of two 16-token passes. Two native
ABBA runs preserve all 96 complete requests. Decode speed is essentially
unchanged. A fresh matched FP32 vLLM-Metal comparison preserves all 48
complete requests, but does not meet every 2x decode or any half-TTFT target.

## Implementation and numerical evidence

`GARNET_METAL_FLOAT_DENSE_KEY_PREFILL=ON` keeps original BF16 checkpoint bits
and decode operations. Dense prefill key projections use FP32 matrix tiles
with 64-term partial sums and a balanced reduction of up to 16 partials.
This removes the large-profile key-cache reference failure without changing
the existing acceptance thresholds. Other projections retain their current
implementation. The option defaults OFF.

The earlier 128-term candidate passes ordinary tests but fails an extended
Qwen3 batch-2, chunk-64 case; its source and failure are saved under
`rejected-128/`. The 64-term candidate passes 144 exact dyadic kernel cases,
all 11 ordinary handoff checks, and 48 independent numerical comparisons:
both models, first/last layers, batches 1/2, chunks 32/64 and three state-carry
steps. These cover inactive rows, partial tails, duplicate/backward slots,
invalid controls, unchanged state after rejected input, reset, closed graphs
and exact resident/explicit agreement. Worst normalized error is 0.992665,
below the existing threshold of 1. The reference uses wider dot accumulation;
NumPy vectorizes acceptance checks while preserving the scalar rules.

`DenseMetalDecoder.enable_prefill(32, 'shared')` and chunk 64 require the
native `dense_key_split_k` capability and hidden size <=1024. Older builds
still reject these profiles before changing decoder state. Twelve actual
old-build requests confirm rejection and preserve their complete outputs.
Default profiles and the MPS projection policy retain their previous limits.

## Repeat native evidence

Each ABBA run uses two fresh processes per policy/profile, three requests
per process and the four non-first requests for warm means. All output IDs,
finished flags and prompt IDs match the frozen prior result. The worker
verifies actual chunk plans `[32]` and `[16,16]`.

| Model | Batch | First run TTFT reduction | Confirmation TTFT reduction | Confirmation decode ratio |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 24.35% | 22.34% | 0.998 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 28.39% | 27.58% | 0.997 |
| Qwen3-0.6B | 2 | 16.26% | 16.60% | 0.999 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 16.99% | 17.25% | 0.999 |

Native confirmation first-use TTFT is approximately 165/223 ms for text
and 137/155 ms for coder, batches 1/2. Initialization is measured separately.
Text batch-2 peak child RSS rises about 54 MiB; other profiles stay near or
below their controls. RSS is not aggregate or unique GPU-memory accounting.

## Fresh matched comparison

Both runtimes use the same M4 / 16 GiB Mac mini, learned values, prompts,
stop rules, output caps and batch sizes, with FP32 operations/activations/KV.
Garnet stores original BF16 weights; vLLM-Metal uses the previously verified
lossless FP32-expanded checkpoints. Exact-value proofs and asset hashes are
checked before execution. This is the additional FP32 study:
`counts_as_default_bf16_speedup=false`. Stock BF16 results remain separate.

| Model | Batch | Garnet tokens/s/sequence | vLLM tokens/s/sequence | Decode ratio | Garnet TTFT ms | vLLM TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 1 | 55.97 | 37.05 | 1.51 | 54.34 | 86.31 |
| Coder | 1 | 71.68 | 46.54 | 1.54 | 39.73 | 62.26 |
| Text | 2 | 51.51 | 26.42 | 1.95 | 97.45 | 97.75 |
| Coder | 2 | 69.49 | 29.29 | 2.37 | 61.54 | 84.31 |

Four warm samples per runtime/profile are averaged. Prefix reuse and
speculation are disabled; runtimes run sequentially and shut down between
workers. The unchanged WorldSense service is allowed only after source/
launcher fingerprints and loaded libraries establish that it runs no
inference. Commands, terminal statuses, process checks, clocks, phase
counters, initialization, first use and RSS are retained under `completed/`.

The measured candidate library SHA256 is
`c0917b8369f89dde5b7688eb3df800905822aba52b1d0d5562f322f5cab2f36d`.
The prior selected library remains unchanged at
`26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671`.
The completed evidence archive SHA256 is
`83a33ccb27d54ececc9c04a9e58b71c5afa0b5febe83fb733ecd69211da87829`.

The result is limited to the actual short prompts and batches measured;
text/coder answers contain 8/17 tokens including stop. VL was not changed
or re-benchmarked in this dense comparison. Its buffer-copy experiment is
separate and is not counted as an achieved speedup here. The full three-model
2x/half-TTFT goal remains active.

## Startup intervals in the fresh matched comparison

Filesystem/driver caches were not flushed. These intervals are excluded from warm means.

| Model | Batch | Runtime | Initialization s | First-use TTFT ms |
| --- | ---: | --- | ---: | ---: |
| Qwen3-0.6B | 1 | garnet | 1.86 | 172.27 |
| Qwen3-0.6B | 1 | vllm-metal | 18.32 | 65.94 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | garnet | 2.12 | 137.89 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | vllm-metal | 16.49 | 62.10 |
| Qwen3-0.6B | 2 | garnet | 2.55 | 208.07 |
| Qwen3-0.6B | 2 | vllm-metal | 16.88 | 105.47 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | garnet | 2.13 | 161.45 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | vllm-metal | 16.34 | 94.79 |
