# Opt-in finer dense prefill profiles

Dense Qwen can select 24/48-token profiles to reduce padding, with
`enable_prefill(32, 'shared', finer_chunks=True)`. This requires a resident
split-key Metal decoder and hidden size <= 1024. The option defaults false;
existing Qwen3.5 plans remain unchanged. No kernel or learned value changed
in this experiment. Both native A/B policies use the same small-output
decode library, SHA256 `56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4`.

The 22-token matched text prompt selects 24 instead of 32 slots. The coder's
26-token prompt still selects 32. Independent NumPy validation passes all
48 comparisons for 24/48 profiles, both real checkpoints, first/last layers,
batches 1/2, three carried-state steps, inactive slots and overwritten slots.
Worst tolerance ratio is 0.992664862 (<= 1 passes), without widening bounds.
These checks use the retained split-key/linear-delta library; the only later
library change affects one/two-row decode. Whole-request execution uses
the small-output library above.

CPU-only planning/admission checks pass 4,096 cases. Actual legacy backends
reject 24/32/48/64 and finer scheduling without changing decoder state, then
complete 12 text/coder requests with frozen exact outputs. Six fresh-image
VL requests retain their previous chunk plans and exact complete captions.
Another 24 complete longer-prompt requests (42 text / 44 coder prompt IDs)
match opt-in 48-slot outputs against 64-slot controls at batches 1/2.
This longer-prompt check is separate from the matched comparator workload.

Two sequential native ABBA runs preserve all 96 complete requests. Each
policy/profile uses two fresh processes with three requests; four non-first
requests form the warm mean. Prompt IDs, output caps, stop rules and complete
output IDs are unchanged. Original BF16 storage and FP32 arithmetic remain.

| Model | Batch | Initial TTFT reduction | Confirmation TTFT reduction | Confirmation decode ratio |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 3.8% | 6.8% | 0.995 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.2% | -0.3% | 1.001 |
| Qwen3-0.6B | 2 | 5.4% | 6.1% | 0.997 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | -0.6% | 1.6% | 1.003 |

Text warm TTFT improves modestly in both runs; decode is essentially flat.
Coder has the same plan, and its small timing changes do not establish a
benefit. Initialization, first-use TTFT, peak child RSS and GPU phases remain
in the raw records. Initial text batch-1 initialization is 2.46 s versus
1.65 s; no unverified cause is assigned. Filesystem/driver caches were not
flushed, so these fresh-process samples are not guaranteed cold hardware.
Recorded wait time overlaps GPU execution and must not be added to it.

## Fresh equivalent-FP32 comparator study

This additional study compares native original BF16 weight storage with
FP32 arithmetic against losslessly expanded, bit-equivalent learned values
in vLLM-Metal FP32 weights/activations/KV. It does not replace or establish a
default BF16 speedup. Actual output/prompt parity, stop-token accounting and
clean shutdown pass for all 48 requests. Inference is sequential on the same
M4/16 GiB Mac mini; prefix reuse and speculation are disabled.

| Model | Batch | Garnet TTFT ms | vLLM TTFT ms | Garnet steps/s | vLLM steps/s | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 51.20 | 85.36 | 56.17 | 37.23 | 1.509 | 0.600 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 41.45 | 62.84 | 72.41 | 46.96 | 1.542 | 0.660 |
| Qwen3-0.6B | 2 | 91.93 | 97.83 | 51.65 | 26.81 | 1.927 | 0.940 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 62.61 | 83.58 | 70.56 | 29.53 | 2.389 | 0.749 |

Steps/s is per synchronous decode step (per row at batch 2); aggregate batch-2
output throughput is twice this. Warm timing excludes initialization. No
new VL vLLM comparison ran here; six VL requests above check compatibility.

The full three-model 2x decode / half-TTFT goal remains active and unmet.
Only these measured prompts/profiles are established. The smaller padded
profile is experimental and opt-in; this is not a general production claim.

`completed/` contains source snapshots, actual commands, terminal statuses,
output IDs, timing/resource samples, guard evidence and checkpoint-equivalence
proofs. The executed guard's LF copy is preserved; its committed copy retains
the repository's CRLF style with identical Python source. Private tensor/image
bytes remain on the Mac with their hashes in `evidence-proof.json`.
Archive SHA256: `25cbd15b85d9966c367c3386d15cd30421cfe54c3efdabb7f73bcf079f58b6ea`.
