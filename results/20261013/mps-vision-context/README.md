# Batched MPS vision value projection: rejected

This prototype extends the selected MPS Q/K score path with batched FP32
probabilities × values. A packing kernel isolates heads/batches and a second
kernel restores the normal hidden layout. It reuses the score path's bounded
scratch, preserves checkpoint values and leaves decoder arithmetic unchanged.

It builds and passes all eleven handoff jobs. The added 48 FP64 value-projection
references cover 4/24/216 patch tokens, batch 1/2, heads 2/12, dimension 8/64,
changed input, abort recovery, and scores/context encoded in the same command
batch while reusing scratch. Actual pretrained vision and all three decoder/
head references pass. All 24 real-frame requests produce identical complete
token IDs to the selected MPS-score control.

| Batch | Control warm TTFT | Candidate warm TTFT | TTFT reduction | Control tokens/s/row | Candidate tokens/s/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 545.79 ms | 542.48 ms | 0.61% | 47.02 | 47.17 |
| 2 | 759.77 ms | 756.31 ms | 0.46% | 42.94 | 43.12 |

Same M4 Mac mini / 16 GiB, Qwen3.5-0.8B, 288×192 image, 54 visual tokens,
79 prompt IDs, chunk 16, 32-token cap, fresh image encoding and decoder reset.
Twelve requests per batch use sequential native ABBA ordering; first-use
requests are excluded from warm means. The sub-1% changes do not demonstrate
a useful full-model gain or approach the 2× goal. No extra confirmation or
new vLLM speed claim is warranted. The three prototype source changes were
reverted after archiving; the validated score-only implementation is kept.

The control library hash is
`c9f8c76aa2657669d6a7b10eba459481ac797fc64904d25360933b565f9d77d1`.
`candidate-source/` retains the prototype, flags, actual build/check commands,
CTest detail and hashes. Job folders retain terminal exits, process guards,
all frame timings/token IDs and identical image/patch hashes. Raw image
buffers are excluded. All overall goal targets remain unmet.
