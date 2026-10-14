# Selected attention policy versus vLLM-Metal

The full goal remains unmet. A fresh sequential comparison completed three
actual GPU dtype audits, all six profiles, and 72 complete matching outputs.
All batch-2 profiles exceed 2x decode in this run; no batch-1 profile does.
No warm first-token latency reaches half of vLLM-Metal's latency.

## Warm measurements

Means of four warm requests per runtime/profile, drawn from two fresh workers.
Each worker's first request is excluded. TPS below is per row; aggregate TPS
at batch 2 is twice the displayed value because the identical rows finish
together. The first prediction is excluded from decode TPS; terminal
predictions count. TTFT includes the measured request preparation and prefill.

| Model | Batch | Garnet TPS | vLLM TPS | Decode ratio | Garnet TTFT ms | vLLM TTFT ms | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 59.89 | 37.02 | 1.618 | 51.27 | 83.50 | 0.614 |
| Qwen3-0.6B | 2 | 58.62 | 28.25 | 2.075 | 86.19 | 89.81 | 0.960 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.07 | 47.19 | 1.527 | 36.83 | 58.01 | 0.635 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.70 | 28.60 | 2.472 | 56.60 | 80.16 | 0.706 |
| Qwen3.5-0.8B VL | 1 | 47.73 | 26.16 | 1.824 | 210.79 | 265.35 | 0.794 |
| Qwen3.5-0.8B VL | 2 | 44.26 | 17.34 | 2.552 | 390.23 | 438.08 | 0.891 |

Batch-1 decode still needs another 23.6% text, 30.9% coder, and 9.6% VL
speedup relative to these measurements. TTFT needs another 18.6–47.9%
reduction across the six profiles. Exact targets, aggregate TPS and separate
setup observations are in [remaining-budgets.json](remaining-budgets.json).
These measurements do not justify a completion claim or a favorable-number
repeat without a new change or an unresolved measurement concern.

The [native phase summary](native-phases.json) points to additional work
outside the measured prefill session: text B2 TTFT is 86.19 ms while the
session is 62.62 ms; text B1 is 51.27 versus 43.83 ms. This difference includes
reset, embedding assembly and wrapper work and does not attribute a cause.
Source inspection finds that resident reset creates a temporary zero-filled
CPU vector for each state and copies it into the existing shared buffer.
A bounded direct-buffer zeroing experiment is warranted; its benefit remains
unmeasured. Decode is mainly GPU work in these observations, so this experiment
cannot establish the missing decode gains. GPU and wait times overlap and
must not be added.

## Initialization and first use

These are separate measured intervals, averaged over two fresh workers.
First-request TTFT starts after initialization. OS/driver caches were not
flushed; this is not a hardware-cold end-to-end latency measurement. The
intervals are not added and presented as an observed cold TTFT.

| Model | Batch | Garnet init s | vLLM init s | Garnet first-request TTFT ms | vLLM first-request TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 1 | 1.690 | 17.212 | 158.22 | 68.19 |
| Text | 2 | 1.884 | 16.377 | 180.19 | 107.51 |
| Coder | 1 | 1.383 | 15.964 | 119.96 | 60.46 |
| Coder | 2 | 1.391 | 16.004 | 146.03 | 96.45 |
| VL | 1 | 4.069 | 33.515 | 357.81 | 302.29 |
| VL | 2 | 4.795 | 33.150 | 571.70 | 506.80 |

Garnet initializes faster, but its first request after setup is slower in
every profile. Warm speed must not obscure this limitation.

## Scope and selected builds

This is an additional matched-FP32 study. Learned values, prompts, image
bytes, capacities, output caps, stopping rules, activations, arithmetic and
KV precision match. Garnet retains compact BF16 weight storage and exact
scaled-half prefill storage; vLLM loads losslessly expanded FP32 weights.
The weight widths and memory traffic differ. This study does not isolate
the contribution of that difference or prove a win over stock BF16 vLLM.

The optional contiguous-attention candidate is selected for text/coder B1/B2
and VL B1 after independent native confirmation. VL B2 uses the retained
control because its native gain was inconclusive. The primary serving
library is unchanged and the experiment remains default OFF.

- Candidate SHA256: `f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088`.
- Control SHA256: `73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
- Native plans: text B1 `[24]`, B2 `[22]`; coder `[26]`; VL B1 `[80]`, B2 `[32,32,16]`.

The fixed source image is `RealWorld-04.png` (WEBP bytes), RGB 1818 x 1372,
prepared at the selected 288 x 192 profile with 216 patches, 54 visual tokens,
79 prompt IDs and a 32-token output cap. Images are freshly encoded per
request. Prefix/encoder reuse and speculation are disabled. Runtimes run
sequentially under a conflict guard and a 2 GiB free-space guard. Small models,
short frozen prompts, one image and equal batch rows limit generalization;
larger models, long contexts, diverse batching and broad task quality remain
unproven.

Hardware: M4 Mac mini, 10 CPU cores (4 performance/6 efficiency), 10 GPU
cores, 16 GB unified memory, macOS 15.6.1 ARM64. Installed comparator versions:
vLLM `0.30.0+cpu` using the verified MLX GPU backend, vLLM-Metal `0.30.0`,
MLX `0.32.1`, mlx-lm `0.32.0`, mlx-vlm `0.6.17`, torch `2.13.0`,
transformers `5.19.0`, NumPy `2.3.5`, Pillow `12.3.0`. Dependencies remain
inside the isolated comparator environment.

## Reproduction and evidence

Remote root: `/Users/zsdx/garnet-metal-investigation`. The completed runner
was `run_selected_attention_vllm.py`, with logs under
`logs/selected-attention-vllm-20261013`. Its archived executed source records
the exact sequential subprocess commands, environment policy and guards.
The selected policy, checkpoint equivalence proofs, actual dtype audits,
source/library hashes, complete outputs, monitors, versions and exit codes
are retained in [evidence](evidence/).

Archive SHA256:
`7dcc7ab02498e1d91ab726a1a1f6997afceadbac5224b3964793c70804c60d68`.
The verifier checks all 414 archived file hashes, three actual FP32 GPU
audits, six profiles and 72 complete matched outputs, plus build policy,
prompt/chunk plans, fresh encoder identities and timing calculations.

```text
python test2026/metal/verify_selected_attention_vllm.py results/20261013/selected-attention-vllm
python test2026/metal/summarize_selected_attention_vllm.py results/20261013/selected-attention-vllm
python test2026/metal/summarize_selected_attention_phases.py results/20261013/selected-attention-vllm
```
