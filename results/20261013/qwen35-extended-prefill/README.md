# Extended Qwen3.5 prefill: small batch-1 latency gain

GARNET_METAL_QWEN35_EXTENDED_PREFILL=ON admits up to 80 prompt slots, with
an 80*1024 packed-hidden bound. The flag defaults OFF and requires retained
convolution/delta chunk kernels. The GPU shader/arithmetic are unchanged;
the C++ profile bounds and Python admission change. Default bounds remain
64 slots / 65536 packed hidden. Other model profiles remain unchanged.

MetalDecoder.enable_prefill(80, 'shared', linear_chunk=True) requires a
resident session, hidden size <= 1024 and batch*chunk <= 128. It turns this
79-token image prompt into [80] instead of [64,16] at batch 1. Two [40,40]
chunks were tested at batch 2 against [32,32,16]; they show no benefit and
are not recommended for this workload. Keep maximum 32 at batch 2.

All 36 independent real-weight comparisons pass for first/last linear and
full-attention layers, chunks 40/80 (batch 1) and 40 (batch 2), three carried
state steps, inactive tokens and overwritten slots. The original tolerance
5e-6 + 5e-5*abs(reference) is unchanged. Explicit and resident GPU results
also match exactly. Invalid controls, reset and closed-graph rejection pass.
Worst numerical tolerance ratio is 0.599987384 (<= 1 passes).

All eleven standard kernel/serving/numerical/head/vision tests pass. CPU-only
planning/admission checks pass 4,096 existing dense cases and 768 extended
Qwen3.5 cases. Actual candidate and legacy workers reject invalid profiles
without changing profile/cache/slot state, then complete image requests.

Two sequential native ABBA runs retain identical complete captions, prompt
IDs and fresh RGB/patch bytes across all 48 timed requests, with the same
32-token output cap and 28 generated IDs including stop. Each build/batch
uses two fresh workers with three requests; four non-first requests form
the warm mean. No prefix/encoder reuse or speculation is introduced.

| Batch | Initial TTFT reduction | Repeat TTFT reduction | Repeat candidate/control ms | Repeat decode ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.7% | 4.0% | 217.38 / 226.45 | 1.002 |
| 2 | -1.0% | -0.2% | 384.03 / 383.09 | 1.006 |

The batch-1 latency gain is small and repeats; decode is effectively flat.
Batch 2 is slightly slower in both runs. Only the batch-1 80-slot policy is
selected for subsequent comparison. The 40-slot profile remains experimental.
First-use and initialization are separate from warm timing; caches were not
flushed. Peak process RSS includes initialization and is not unique GPU memory.

| Batch | Repeat first-use TTFT candidate/control ms | Repeat initialization candidate/control s |
| --- | ---: | ---: |
| 1 | 342.34 / 429.02 | 3.07 / 3.08 |
| 2 | 509.55 / 573.42 | 3.08 / 3.08 |

The first 80-slot test exposed the prior packed-hidden ceiling; its failure,
initial source/binary hash and retained fixture are preserved. Test cleanup
is fixed so an unsuccessful build is not masked by a session-reset error.
The corrected profile passes without altering numerical tolerances.

The WorldSense supervisor changed during validation. Read-only import/literal
audit confirms it only forwards GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB and
loads no inference libraries. Only its exact inspected source fingerprint
6ce05f32376832f579e5908399a9787d5957cc1b08bff31b6394c09fff9b80ec is added;
the existing identity/loaded-library rejections remain. All 21 guard checks
pass against the live service. Only the investigation's old preflight waiter
was stopped to load the corrected guard, before any test started. The successful
36-case numerical job is reused. External services/processes were unchanged.
Failed dependencies and the stopped preflight are distinguished from executed
numerical or benchmark failures in their saved statuses.

No 2x decode / half-TTFT achievement is established. The full three-model
goal remains active and unmet; these profiles are verified only on the
M4/16 GiB Mac mini, this checkpoint, image/prompt and batches 1/2.

Completed source, commands, statuses, outputs, profile plans, guards,
resource samples and failed/retried evidence are in completed/. Private
tensor/image bytes and binaries remain on the Mac with hashes in
evidence-proof.json. The archive verifier's first version misclassified
both native worker labels as candidates; the corrected verifier uses folder
identity and verifies each executed plan. Benchmark records were unchanged.

Candidate library SHA256: cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8.

Unchanged control library SHA256: 56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4.

Native evidence archive SHA256: 0ef070c976813fec8570645a85e0af7b2f4e9c3d1068cc37da5f8a8897a1ac3c.

## Fresh equivalent-FP32 comparator study

The subsequent comparison uses chunk 80 at batch 1 and retains chunk 32 at
batch 2. Original learned values, source RGB, prompt IDs and output cap are
matched, with sequential native/vLLM-Metal workers and no reuse/speculation.
All 24 requests pass exact output and prompt parity and fresh-encoder checks.

This additional study compares native original BF16 storage with FP32
arithmetic against losslessly expanded, bit-equivalent FP32 comparator
weights/activations/KV. It does not establish stock/default BF16 speedups.

| Batch | Garnet TTFT ms | vLLM TTFT ms | Garnet steps/s | vLLM steps/s | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 221.69 | 256.38 | 48.45 | 26.75 | 1.811 | 0.865 |
| 2 | 385.08 | 447.87 | 44.91 | 16.44 | 2.732 | 0.860 |

Steps/s is per synchronous step/per row; aggregate batch-2 token throughput
is twice this. Neither batch achieves half-TTFT, and batch 1 misses 2x decode.
No new text/coder comparator ran here. The full goal remains unmet.

Comparator commands, terminal statuses, parameter-equivalence proof,
source/asset hashes and raw request records are in comparator/.
Comparator archive SHA256: 52e0ee17976721055b44e70e3012041e82bdc1c563b29600cce861b491c4d4eb.
