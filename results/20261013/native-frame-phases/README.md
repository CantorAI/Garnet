# Actual warm frame phase attribution

The selected MPS-score library runs twelve actual Qwen3.5-0.8B image requests
on the M4 / 16 GiB. All complete token IDs match the frozen control. Each
batch uses two fresh persistent workers with three requests each; the four
warm requests per batch exclude first use. Model, image bytes, 79 prompt
tokens, 54 visual tokens, chunk 16 and output cap 32 are unchanged.

Counters come from the existing native decoder session. No per-operation
GPU serialization or diagnostic synchronization is enabled. Metadata is
emitted after the existing first/done markers. The added assembly timestamp
and metadata do not change learned math or output handling.

| Batch | End-to-end TTFT | Vision/reset | Prompt assembly | Prefill wall | CPU prepare | CPU encode | GPU execution | Submit/wait wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 546.30 ms | 61.83 ms | 0.30 ms | 430.81 ms | 0.97 ms | 48.22 ms | 374.21 ms | 380.31 ms |
| 2 | 761.25 ms | 126.90 ms | 0.36 ms | 580.16 ms | 1.07 ms | 48.64 ms | 522.82 ms | 528.83 ms |

End-to-end timing also includes common CPU image preprocessing and worker
protocol. GPU execution is inside submit/wait wall, not an additive stage.
Prefill counters exclude model/graph first-use compilation from warm means.
The GPU dominates language prefill; removing CPU encoding alone would not
reach the half-TTFT target. This motivates testing more parallel recurrent
dot reductions rather than optimizing the already small prompt assembly.

`native-frame-phases-20261013/jobs/summary.json` contains all phase counters,
request timing, token IDs and input identity. Actual commands/exits, process
guards and worker source hashes are retained. The selected library hash is
`c9f8c76aa2657669d6a7b10eba459481ac797fc64904d25360933b565f9d77d1`.
These measurements do not establish the overall 2×/half-TTFT goal.
