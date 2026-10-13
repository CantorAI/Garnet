# Parallel delta GPU prototypes: rejected

Two isolated GPU implementations of the validated FP32 chunk formulation
were executed sequentially on the M4/16 GiB Mac mini. Both are slower than
the retained gated_delta_chunk kernel, so neither is integrated into serving.
The selected extended-prefill library remains byte-identical:
cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8.

The first implementation uses six batched MPS FP32 products and a direct
forward substitution for the triangular recurrence. The second replaces
those products with 16x32 output tiles and 32-wide reduction tiles in Metal
SIMD-group matrix operations. Both preserve direct decay products without
clipping or division, including zero-decay and inactive tokens.

Both pass 68 output/state comparisons against the retained GPU kernel with
the unchanged 5e-6 + 5e-5*abs(reference) tolerance. The worst tolerance ratio
is 0.039274359 in each run. There are 64 synthetic cases covering batches
1/2, heads 2/16, chunks 16/32/64/80, inactive tokens, correlated keys and
zero decay. Four cases use actual trained layer 0/22 projections with
synthetic hidden inputs and prior states. These are core component checks;
they are not complete layer, model, image-caption or vLLM comparisons.

Each case runs two ABBA timing rounds, three warmups per position, and twelve
measurements per position. The table is the median GPU command-buffer time
over the four positions for each implementation. Control timings are shown
separately for each run. Encoders and matrix operations execute in one command
buffer; wall time is also retained in raw logs. Guard snapshots before and
after each run check for competing inference.

| Layer / batch / chunk | MPS-run control ms | MPS candidate ms | Tiled-run control ms | Tiled candidate ms |
| --- | ---: | ---: | ---: | ---: |
| 0 / 1 / 80 | 0.2776 | 0.8550 | 0.2799 | 1.0710 |
| 0 / 2 / 32 | 0.2101 | 0.6000 | 0.2137 | 0.7190 |
| 22 / 1 / 80 | 0.2787 | 0.8504 | 0.2849 | 1.1446 |
| 22 / 2 / 32 | 0.2100 | 0.5995 | 0.2084 | 0.7112 |

MPS is 2.85-3.08x slower; tiled Metal is 3.36-4.02x slower in these trained
projection cases. The additional matrix work and dispatches are a plausible
cost, but this experiment does not isolate each operation's contribution.
There is no useful speed gain to justify full-model integration or a repeat
run solely to seek a favorable sample.

Commands, terminal statuses, source hashes, assets manifests, raw numerical
checks, timing windows and summaries are in completed/. The exact standalone
sources are in source/ and test2026/metal/. Input tensors and compiled binaries
remain on the Mac; their byte sizes and hashes are in evidence-proof.json.
The transferred archive SHA256 is
78bdeb4488a408b5e8e83b88bc6f7b4dbdd57a0e591f00ef213acdb8263328b9.

The full goal of 2x vLLM-Metal decode and half first-token latency across the
three matched Qwen workloads remains active and unmet. These rejected probes
provide no end-to-end improvement and no new comparator results.
