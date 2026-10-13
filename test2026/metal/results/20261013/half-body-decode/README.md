# Rejected exact-scaled FP16 body decode

The selected build remains `build-kv-chunk-vector`, SHA-256
`564b515ab6b0eb68413ef5a54de6beff47d333f8f1fa6a1188562bf6e5c28ec4`.
The candidate reads the existing exact-scaled FP16 body-weight cache in
decode, accumulates in FP32 using the original lane/input grouping, then
reverses each output-column scale. Large heads and incompatible matrices
keep raw BF16. Weight values, FP32 activations/KV, model files and caps remain
unchanged. The candidate C++ and build flag are reverted locally and remotely.

All 11 native numerical/serving/vision checks pass. The first completed
sequential native ABBA run executes 48 text/coder requests and 24 actual image
requests with matching complete outputs, prompts and finish flags. These
initial ratios are not sufficiently reproducible to select the candidate.

| Model | Batch | Initial decode ratio | Initial TTFT fraction |
| --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 1.008 | 0.988 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.001 | 1.009 |
| Qwen3-0.6B | 2 | 1.061 | 0.900 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.999 | 1.012 |
| Qwen3.5-0.8B, image | 1 | 1.119 | 1.001 |
| Qwen3.5-0.8B, image | 2 | 1.058 | 0.969 |

The initial Qwen3 batch-2 and image controls run materially slower than
the preceding selected-build observations. A clean repeated 48-request
text/coder comparison yields only 0.4–1.0% decode gain, with 0.6–2.1% worse
warm TTFT; all outputs still match. Four warm requests per build/profile
are averaged, with first-use and initialization retained separately.

| Model | Batch | Repeat control tokens/s | Repeat candidate tokens/s | Repeat TTFT fraction |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 56.91 | 57.41 | 1.006 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.29 | 72.58 | 1.017 |
| Qwen3-0.6B | 2 | 52.12 | 52.63 | 1.021 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.07 | 70.34 | 1.015 |

Two confirmation attempts are stopped by the unchanged exclusivity guard
after unrelated release-test xlang3 processes appear: PID 65663 interrupts
the first text repeat, and PID 66625 interrupts the subsequent image repeat.
Incomplete/interrupted samples are not used to claim improvement. PID 65663
is missing on read-only follow-up. Unrelated processes are never terminated
and the guard is not relaxed. Image throughput improvement remains unproven.

The first launch mistakenly calls the previous build helper and rebuilds
the control; missing-candidate checks fail and dependent timing jobs are
gated. The control is restored from committed source and its exact SHA-256
is verified before the corrected candidate build. Those launch/restoration
logs remain intact. No timed run uses the accidentally rebuilt control.

The candidate is rejected: repeatable benefit is too small, TTFT regresses
slightly, and larger initial/image gains are not established. There is no
new vLLM benchmark or 2x/half-TTFT claim. The full objective stays active
and unmet. Recorded commands/exits, source/build/model/input hashes, actual
outputs, numerical references, guards and interrupted confirmations are
saved; raw RGB/patch arrays and binaries remain private on the Mac.
Protected GPT-OSS, Pi and original Mac checkouts are untouched.
