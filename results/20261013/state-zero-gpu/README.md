# GPU buffer-fill reset: rejected component variant

A synchronous Metal blit fill is slower than direct CPU clearing in all six
model/batch reset profiles, both with CPU-warmed buffers and immediately
after GPU writes. Do not integrate this variant into serving. No native
library, model precision or checkpoint changed in this experiment. The full
2x-decode/half-TTFT goal remains unmet.

## Measured reset wall time

State sizes come from the actual three checkpoint configurations at capacity
256 and batches 1/2, including full-attention KV and VL convolution/recurrent
state. Allocations use the same MPS FP32 row-stride calculation and shared
Metal storage as the native context. These are synthetic reset components,
not full model requests; their cache/resource behavior differs from serving.

Each run uses two rounds of policy order legacy/direct/GPU/GPU/direct/legacy,
three warm operations per window and eight measured operations per window.
Reported values are medians of the four window medians for each policy.
GPU reset timing includes command creation, encoding, submission and waiting
for completion. Source compilation, allocations and correctness readbacks
are excluded on all sides. In the GPU-conditioned run, a compute kernel
writes every logical state element and completes before each warm/sample
reset; that conditioning is excluded from reset time, as preceding work.

| Model | Batch | Logical MiB | CPU-warmed direct CPU ms | CPU-warmed GPU fill ms | After GPU writes: direct CPU ms | After GPU writes: GPU fill ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 56 | 0.510 | 1.062 | 0.846 | 1.364 |
| Qwen3-0.6B | 2 | 112 | 1.750 | 2.192 | 1.353 | 1.866 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 6 | 0.081 | 0.656 | 0.053 | 0.329 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 12 | 0.110 | 0.692 | 0.196 | 0.740 |
| Qwen3.5-0.8B | 1 | 25.6875 | 0.409 | 1.262 | 0.235 | 0.531 |
| Qwen3.5-0.8B | 2 | 51.375 | 0.790 | 0.959 | 0.832 | 1.073 |

The original temporary-vector/copy control is also retained in raw results.
For example, its text B2 component times are 22.094/21.039 ms in these runs.
The padded 1,032-byte profile validates logical-element clearing without
overwriting row padding; it is not a serving performance profile.
GPU fill is about 1.29–6.24x slower than direct CPU clear in the six
GPU-conditioned profiles. The raw summary field `gpu_over_direct_cpu` means
CPU-time/GPU-time speedup; values below one indicate a slower GPU variant.

Both conditions reject the measured synchronous blit design. This does not
reject every possible GPU reset implementation or predict exact full-model
latency. In particular, the isolated direct-clear text B2 time is lower than
the 5.1 ms observed in the actual native confirmation. Buffer/driver/cache
behavior and surrounding work differ; component times must not be substituted
into full-model TTFT or mixed with vLLM throughput measurements.

## Validation and retained failures

Two terminal successful runs perform 126 nonzero-to-positive-zero checks,
verified by CPU bit inspection and a GPU verification kernel, with row padding
preserved. Checks use three different nonzero CPU patterns per policy/profile;
GPU conditioning applies to the timed windows. Seven profiles per run cover
the six model states and the deliberately padded case. There are 168 timing
windows, each with eight measured operations. Conflict guards reject other
inference/probe processes and enforce at least 2 GiB free space.

The first build failed because of a missing closing brace before any GPU
timing. Its exact source, compiler output, command and exit 1 are retained.
The corrected CPU-warmed run and the additional GPU-conditioned run exit 0.
No observation timeout caused a restart; no unrelated job was stopped.

Owned root: `/Users/zsdx/garnet-metal-investigation`. Durable jobs:
`state-zero-gpu-probe-20261013` (PID 30139, failed compile),
`state-zero-gpu-probe-retry1-20261013` (PID 30289, success), and
`state-zero-gpu-probe-conditioned-20261013` (PID 30560, success).
All are terminal. No additional inference dependencies were installed.

Apple's [buffer-fill API documentation](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/fill%28buffer%3Arange%3Avalue%3A%29)
defines the byte-fill operation used by the prototype. Synchronous completion
and the real zero/padding checks are exercised on the M4, not inferred from
the API's availability.

## Reproduction and evidence

The command is `run_state_zero_gpu_probe.py ROOT NEW_OUTPUT_DIR`, with
`--gpu-conditioned` for the second condition. Exact absolute runner/compiler/
GPU commands, metadata/configurations, source hashes and snapshots, profiles,
all checks/window timings, monitors and exits are in [evidence](evidence/).

```text
python test2026/metal/verify_state_zero_gpu.py results/20261013/state-zero-gpu
```

The verifier checks 48 archived file hashes, all three source snapshots,
terminal statuses, actual profile sizes, the conditioning command, all 126
checks and timing calculations. Archive SHA256:
`9a4cc42d3746741869cb27ba227caa6fe660f23516bcc4a2bea910b3513a50d2`.
Primary libraries and GPT-OSS/Pi checkouts remain unchanged. The retained
[direct CPU reset experiment](../direct-state-zero/README.md) still has its
documented latency/decode tradeoff. Prefill and decode optimization remain
necessary; the [latest vLLM comparison](../selected-attention-vllm/README.md)
continues to record the remaining targets across all six profiles.
