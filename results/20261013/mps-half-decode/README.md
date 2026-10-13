# Rejected mixed-input MPS decode

The candidate extends the existing lossless scaled-FP16 weight cache to
single/two-row MPS multiplication, retaining FP32 activations/output and
original learned values. Incompatible matrices and large heads keep raw
BF16. This is a different route from the earlier custom FP16 body kernel.

All 11 numerical/serving/vision checks pass. Sequential native ABBA runs
preserve every complete output, prompt ID, finished flag and image input
across 48 text/coder and 24 VL requests. Four warm samples per build/profile
are averaged; first use, initialization and memory remain separate.

| Model | Batch | Retained tokens/s/sequence | Candidate tokens/s/sequence | Decode ratio |
| --- | ---: | ---: | ---: | ---: |
| Text | 1 | 56.30 | 45.79 | 0.813 |
| Coder | 1 | 71.96 | 57.12 | 0.794 |
| Text | 2 | 51.79 | 42.09 | 0.813 |
| Coder | 2 | 69.52 | 55.33 | 0.796 |
| VL | 1 | 48.27 | 40.72 | 0.844 |
| VL | 2 | 44.79 | 37.49 | 0.837 |

The candidate is rejected: decode is 16-21% slower across every profile.
No positive speed claim or new vLLM comparison is made from it. Text/coder
TTFT is lower in this run, but complete-answer latency regresses. VL TTFT
is approximately unchanged. The full 2x/half-TTFT goal remains unmet.

The initial preflight stopped on a live service installer. The isolated CPU
build proceeded, while GPU tests waited 97.7 seconds for the installer to
exit and the unchanged exclusivity guard to pass. No external process was
stopped or exempted. Every numerical test has an additional pre-task guard.

Candidate source/flags, commands, terminal exits, complete outputs,
resource/process samples, startup timings and guard wait are preserved in
`rejected/`. Tensor/RGB copies remain private, with hashes and sizes in
`evidence-proof.json`. The archive SHA256 is
`5a29051eeecc6e1c4e161bed0b2173ef1764f749640ea44e5f9cd4aeb8f4ad42`.
Candidate library SHA256 is
`dc3bf94caedd881d0fbc1ef668eef9f2ee47988f28ca0918692239aaf2c1a8a7`.

The two experimental source edits and build option are reverted locally and
on the Mac to the exact prior source. The retained library is not rebuilt;
its SHA256 remains
`e727c3c81e7af9f252bbd090ea506e6db6c2ffee021f2edf6e167e380d9ea5e8`.
Protected GPT-OSS, Pi and original Mac checkouts are untouched.
