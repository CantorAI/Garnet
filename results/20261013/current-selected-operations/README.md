# Refreshed operation attribution for the selected implementation

Six sequential real native workloads cover text, coder and Qwen3.5 VL at
batches 1/2. All complete token IDs, prompt IDs and finished flags match the
latest retained equivalent-FP32 comparison's native outputs. The library is
the selected extended-prefill build, SHA256
cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8.
Text uses the finer 24-slot tail policy, coder 32, and VL 80 at batch 1 / 32
at batch 2. Prompt lengths, image bytes, stop rules and output caps are unchanged.

The diagnostic enables GARNET_METAL_PROFILE_OPS=1, which serializes operations.
Its 60,816 operation records identify costs, but their times must not be
presented as serving performance, added to warm TTFT, or compared to vLLM
tokens/sec. There is one actual request per profile, including first-use;
this is not a cold/warm throughput study. Before/after process guards reject
competing inference. External services and existing checkouts are untouched.

| Model / batch | Decode-width up projections, aggregate GPU ms | Vocabulary head, aggregate GPU ms | Down projections, aggregate GPU ms |
| --- | ---: | ---: | ---: |
| Text / 1 | 29.07 | 23.38 | 14.51 |
| Text / 2 | 28.83 | 23.56 | 15.68 |
| Coder / 1 | 73.29 | 45.29 | 35.86 |
| Coder / 2 | 73.87 | 45.59 | 36.71 |
| VL / 1 | 106.35 | 137.14 | 55.25 |
| VL / 2 | 106.50 | 138.06 | 57.19 |

The table groups recorded operations by shape; it includes the final prefill
head and all decode heads. Aggregates across models have different token and
layer counts. VL has other substantial QKV projections omitted from this
compact table. Exact shapes/counts and the full ranking are in summary.json.

The largest batched prefill up-projection groups cost 12.23/11.31 ms for text,
10.68/16.11 ms for coder, and 16.84/23.28 ms for VL (batches 1/2). The next
experiments should address these projections and the substantial head cost;
the rejected parallel-delta algorithms are not supported by this evidence.
These ranks alone do not prove memory bandwidth or any hardware bottleneck.

For perspective, the latest uninstrumented VL comparator records already
saved in qwen35-extended-prefill/comparator have four warm native requests
per batch. Their worker prefill wall means are 107.76/206.14 ms, vision/reset
61.94/123.76 ms, and assembly 0.315/0.398 ms. Prefill GPU means are
99.83/184.56 ms; CPU encoding is 5.82/16.30 ms. Worker TTFT is
170.01/330.30 ms and excludes the common frontend. End-to-end TTFT is the
221.69/385.08 ms reported in that comparison. Wait time overlaps GPU time.
The much older 431/580 ms prefill attribution is no longer representative.

Commands, six terminal statuses, raw operation logs, current guard evidence,
case files, complete outputs, rankings and runner/worker source snapshots are
in completed/ and source/. Fresh image tensor/RGB files are omitted from the
archive; their identity is recorded in the retained request records.
Archive SHA256:
5955939be8329d35a8bedc3f6a5f9194c6237a223f535bbaed578b1a208de4ab.

This diagnostic establishes no new speed gain. The 2x decode / half-TTFT
objective for all three matched Qwen workloads remains active and unmet.
