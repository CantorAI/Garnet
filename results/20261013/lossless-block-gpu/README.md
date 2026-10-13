# Whole-block BF16 packing: rejected

This exact codec stores 128-value blocks as 128 sign/mantissa bytes plus
64 exponent-tag bytes when the exponent span fits four bits. Other blocks
keep their original 256 raw bytes. One 64-bit metadata word holds each block's
offset, exponent base and compact/raw flag. Unlike earlier nibble codecs,
there are no per-value escapes, prefix sums or separate exception reads.
The original BF16 bits are reconstructed before FP32 dot products.

The isolated M4/16 GiB GPU run completes all three original checkpoints,
their complete heads and first-layer MLP down projections, batches 1/2.
All 12 projections are bit-exact. Full CPU/GPU weight reconstruction also
passes for the matrices, all 65,536 BF16 patterns, and a random 4,097-value
tail, including nonfinite payload bits without multiplying those fixtures.
Another 21 local CPU roundtrips cover arbitrary, compact and mixed inputs
and lengths around block/chunk boundaries; cpu-roundtrip-checks.json retains
the cases and codec source hash.

Packing reduces complete-head bytes by about 21.7%, including metadata.
It does not establish process-memory savings: raw and packed buffers are both
resident in this component experiment. Host packing uses bounded 262,144-value
windows. Checkpoint files and serving code are unchanged.

| Model / batch | Raw head GPU ms | Packed head GPU ms | Raw / packed ratio |
| --- | ---: | ---: | ---: |
| Text / 1 | 2.939 | 3.521 | 0.834 |
| Text / 2 | 2.942 | 3.616 | 0.814 |
| Coder / 1 | 2.592 | 3.144 | 0.825 |
| Coder / 2 | 2.610 | 3.105 | 0.841 |
| VL / 1 | 4.869 | 5.996 | 0.812 |
| VL / 2 | 4.877 | 5.960 | 0.818 |

Each profile has four ABBA rounds, three warmups per position, and twelve
samples per position. The table is the median of the eight position medians
per policy. Runs are sequential with before/after inference guards. Packing,
CPU validation and compilation are outside the GPU clock. Complete heads
are all slower, as are the tested down projections; no serving integration
or favorable-sample repeat is justified. These are not model TPS or TTFT.

Commands, statuses, hardware/source/checkpoint/input hashes, guards, CPU/GPU
checks and raw timing windows are in completed/. Source snapshots are in
source/ and test2026/metal/. Raw tensors/binaries remain privately on the Mac
with sizes/hashes in evidence-proof.json. Transferred archive SHA256:
7f0a58986cc4f1ab9b0bc83a48ae4b68996ea446f1cebc6d3cf7cf1f649a4564.

The selected library remains
cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8.
The three-model 2x decode / half-TTFT goal remains active and unmet.
