# Actual comparator precision audit

The installed vLLM-Metal normal text and VL loaders retain checkpoint weight
dtype. Requesting `dtype=float32` alone changes its KV dtype, not all loaded
weights or activations. Six actual loaded-model inspections establish this
on the M4 GPU, without changing comparator math, loaders or security settings.

The final probe inspects both public model parameters and the original
projection modules held privately by paged-attention wrappers. It evaluates
one embedding, first input norm and first attention projection on GPU and
records their actual output dtypes. Each engine closes before the next starts.
These are dtype diagnostics, not inference speed measurements.

| Model | Requested dtype | Actual parameter tensors | KV dtype | Probed activations |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | BF16 | 310 BF16 | BF16 | BF16 |
| Qwen3-0.6B | FP32 | 310 BF16 | FP32 | BF16 |
| Qwen2.5-Coder-0.5B-Instruct | BF16 | 290 BF16 | BF16 | BF16 |
| Qwen2.5-Coder-0.5B-Instruct | FP32 | 290 BF16 | FP32 | BF16 |
| Qwen3.5-0.8B | BF16 | 437 BF16 + 36 FP32 | BF16 | BF16 |
| Qwen3.5-0.8B | FP32 | 437 BF16 + 36 FP32 | FP32 | BF16 |

The loaded parameter counts are 596,049,920 for text, 494,032,768 for coder,
and 852,985,920 for VL. The hybrid model already contains some FP32 state/
gating parameters; this does not make its entire model FP32.

The first probe failed because callable RPC serialization is disabled by
default. Its engine shut down cleanly. Using the supported in-process engine
permits direct inspection without enabling insecure serialization. Initial
successful inspections counted only public parameters and missed private
attention owners; the final six inspections correct that coverage. All
historical commands, exits and source versions are retained and hash-checked.

`comparator-actual-dtypes-complete-20261013/jobs/summary.json` is authoritative.
Each result includes installed versions, actual model/KV/activation dtypes,
GPU device, wrapper owners and representative parameter shapes. The archived
installed loader and wrapper sources explain the observed behavior.

This confirms that existing Garnet FP32 versus stock comparator BF16 results
have a precision difference. A separate exact-value FP32 representation study
is prepared next; it will stay separate from the default BF16 benchmark.
The 2× throughput / half-TTFT goal remains unmet.
