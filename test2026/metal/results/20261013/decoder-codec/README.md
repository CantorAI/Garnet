# Rejected lossless decoder packing

This experiment preserves every BF16 weight bit. It uses 8 sign/mantissa bits,
4 exponent tags, full exponent escapes and per-128-value metadata. The GPU
reconstructs the original bits before FP32 arithmetic. It is not quantization.
The original BF16 buffer is retained for prefill and fallback, and the large
vocabulary head is excluded from packing. The proposed CMake option defaults
off. Both the option and native implementation were reverted after measurement.

## Component evidence

`decoder-codec-gpu-20261013` tested 29 real projection matrices: dense layer 0
from text/coder, plus Qwen3.5 linear layer 0 and full-attention layer 3. Complete
CPU and GPU bit reconstruction passed for every matrix. All 58 batch-1/2
projection comparisons were bit-exact. Four ABBA rounds per case used three
warm operations and 12 measured operations per policy window.

Summed isolated projection-time ratios were 1.93x/1.89x for text batches 1/2,
1.66x/1.67x for coder and 1.91x/1.82x for the combined sampled VL layers.
These tests repeatedly reuse individual matrices. They are not full inference
or a vLLM advantage, and their cache reuse did not transfer to the model runs.

## Actual full-model results

`build-body-codec` compiled with the experimental option enabled. Its native
host packer reconstructs every packed value at weight-load time; numerical
references exercise GPU decoding. All 11 native checks passed, covering trained
dense/hybrid layers, chunks and inactive rows, serving/reset/shared graph state,
complete heads and vision. The native text/coder ABBA suite then executed 48
actual requests; the native real-image ABBA suite executed another 24. Every
prompt, generated token ID and completed-output flag matches the validated
`build-prefill-wide` control. No competing inference runtimes overlapped.

This table averages four warm requests per build/profile from two fresh
processes. Initialization and first-use requests are retained separately.

| Model | Batch | Decode ratio vs control | Control TTFT, ms | Packed TTFT, ms |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 1.051 | 103.26 | 111.15 |
| Qwen3-0.6B | 2 | 1.040 | 173.32 | 182.15 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.027 | 72.89 | 83.80 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.998 | 99.43 | 112.74 |
| Qwen3.5-0.8B, actual image | 1 | 0.982 | 694.27 | 683.76 |
| Qwen3.5-0.8B, actual image | 2 | 0.977 | 1158.57 | 1126.15 |

Text decode gained about 4–5% and coder batch 1 about 3%, while text/coder
TTFT worsened about 5–15%. VL first-token time improved only 1.5–2.8% and its
decode speed regressed 1.8–2.3%. Individual-process peak RSS reached 2.192 GiB
versus 1.555 GiB for text and 1.795 versus 1.263 GiB for coder. This is child
RSS high water, not aggregate/unique GPU memory. Mean batch-1 process
initialization was 4.377 versus 0.984 seconds for text and 3.549 versus 0.775
seconds for coder. Load-time packing contributes to initialization, not warm
request timing. No three-model resident-memory advantage is claimed.

The broad decoder implementation is rejected. The validated wide-prefill source
and CMake file were restored locally and on the Mac; `build-prefill-wide` is
still the selected implementation. The experimental binary remains isolated.
There is no new matched vLLM speed claim from these native-control tests.
The full 2x-throughput/half-TTFT goal remains active and unmet.

## Retained evidence

The complete commands/exits, hardware/model/library/input hashes, process
snapshots, timing windows, numerical logs and output sequences are preserved.
`body-codec-source-20261013` contains the actual experimental native/CMake
sources and build script. `rejected_body_codec.diff` records the reverted change.
`decoder-codec-gpu-20261013/source-snapshot` preserves the separate matrix probe.
Reproducible `.bin`, patch/RGB buffers and compiled binaries are omitted from
this checkout's evidence and retained privately on the Mac. The original
repositories and protected GPT-OSS/Pi worktrees were not modified.
