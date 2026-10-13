# Rejected compute encoder reuse

The experiment reused one explicitly serial native Metal compute encoder
across custom dispatches, with buffer barriers, and closed it before MPS work,
submission or abort. Dynamic threadgroup memory bindings were reset between
operations. Kernel arithmetic, BF16 values, graph outputs and state layouts
were unchanged. Apple documents multiple kernel dispatches in a compute pass
and resource-ordering barriers in its
[compute encoder documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder).

`build-encoder-reuse` passed all 11 existing native checks, plus a dedicated
custom-compute abort/recovery test. The new test aborts a custom add command,
requires the abandoned output to remain unreadable, then verifies a subsequent
batch completes correctly. It also passed on the restored `build-prefill-wide`
implementation. That transaction test is retained; the reuse code is reverted.

The close native ABBA comparison executed 48 text/coder requests and 24 real
image requests. Every prompt, token/caption sequence and completed-output flag
matched the selected wide-prefill control. There was no inference overlap.
The table averages four warm requests per build/profile; initialization and
first-use records are retained separately, with filesystem caches unflushed.

| Model | Batch | Decode ratio vs control | Control TTFT, ms | Reuse TTFT, ms |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 1.000 | 103.00 | 102.52 |
| Qwen3-0.6B | 2 | 1.004 | 173.84 | 173.92 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.002 | 72.07 | 72.15 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1.001 | 100.38 | 100.49 |
| Qwen3.5-0.8B, actual image | 1 | 1.001 | 693.98 | 690.41 |
| Qwen3.5-0.8B, actual image | 2 | 1.003 | 1136.54 | 1132.99 |

No material full-model advantage is established. Warm decode CPU encode time
fell from approximately 0.395–0.420 ms to 0.322–0.336 ms. Native GPU execution
remained approximately 12.7–18.0 ms, with CPU wait roughly 13.1–18.5 ms. These
phase fields are observations within serving steps, not times to add blindly:
GPU/wait can describe overlapping parts of one command's execution. Saving
about 0.08 ms in CPU encoding did not improve the end-to-end decode clock.

The experiment is rejected for integration and its C++ source was restored
locally and on the Mac. `build-prefill-wide` remains selected. There is no new
vLLM comparison or 2x/half-TTFT success claim. The active full objective is unmet;
these results redirect optimization toward GPU execution and data movement.

`encoder-reuse-source-20261013` contains the exact experimental source/build
scripts and the transaction test. `rejected_encoder_reuse.diff` records the
reverted code. Complete commands/exits, source/model/library/input hashes,
numerical logs, process guards, exact outputs, phase records and timing
summaries are retained. Reproducible raw image/patch arrays and compiled
binaries stay private on the Mac. Protected repositories were not changed.
