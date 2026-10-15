# MPSGraph FP32 prefill: actual component results, no broad promotion

The repaired owned PID84919 exited0 in81.7607s. This is an independent
projection component against Garnet's current native common-alpha MPS
prefill. It uses the same M4/16 GiB Mac, original first-layer learned matrices,
physical rows24/44 text,26/52 coder and80/160 VL, and four input patterns.
Graph matmul inputs, product and outputs are FP32, with runtime tensor dtype
assertions. Original numerical bounds are unchanged.

Policy1 supplies original BF16 values losslessly expanded into FP32 weight
storage. Policy2 supplies the current lossless common-scale FP16 weight
storage, explicitly casts it to FP32 before matrix multiplication and applies
the exact power-of-two alpha in FP32. CPU conversion checks every recovered
original learned-value bit. Both policies preserve original FP32 activations;
there is no approximate activation conversion or reduced-precision matmul
requested. The SDK default graph optimization level is1.

Independent audit verifies28 archived files,400 numerical checks,100 separate
first graph executions,100 admitted matrix/batch/policy timing profiles and
6400 timed submissions. All numerical checks pass the original FP64 envelope
and5e-6+5e-5*abs(native) comparison. Four matrices retain their native routes
and are skipped: the two special text/coder keys and two VL matrices without
the control's common-alpha representation. The archived proof independently
derives this admission from hashed raw BF16 tensors using the original integer
converter's rules. Actual checkpoints and every cached raw matrix are rehashed
and checked against checkpoint tensor slices by the executed runner.

Fractions divide sums of candidate component wall means by sums of control
means over the admitted matrices. Values above1 are slower. These are not
weighted by the models' layer counts or end-to-end inference frequencies.

| Model / batch | FP32 weight graph wall fraction | Cast-half weight graph wall fraction |
| --- | ---: | ---: |
| Text / 1 | 1.0035 | 1.4047 |
| Text / 2 | 0.9402 | 1.2458 |
| Coder / 1 | 1.0597 | 1.4443 |
| Coder / 2 | 0.9937 | 1.3580 |
| VL / 1 | 0.9758 | 1.2339 |
| VL / 2 | 0.9575 | 1.1556 |

The cast-half policy is slower in all groups. The FP32-storage policy has a
small2–6% signal for text B2 and VL, but doubles weight storage/traffic and
does not improve text/coder B1. It does not justify broadly replacing native
prefill. Full-model memory and latency gains have not been demonstrated.
No selected serving library or production source is changed.

Each native product and graph product runs sequentially and synchronizes
before the next. There is one projection per submission; this is not a whole
layer graph or fused MLP. Timings are wall latency only: native command-buffer
GPU span is never substituted for graph GPU time. Allocations/uploads and
first graph execution are excluded from warm windows. Each admitted profile
has two ABBA rounds, two warmups and eight samples per position, with250ms
common warmup. First graph execution, which includes compile/setup and GPU
work, spans3.13–159.11ms (median32.27ms); it is recorded separately, not added
to an invented hardware-cold result.

The initial owned PID84671 failed compilation in6.8532s because an execution
descriptor property was assigned to MPSGraph. No inference ran. Its17-file
archive is preserved under `failed/`; the SDK-default optimized synchronous
call repaired the API use only after that process was terminal and archived.

Repaired archive SHA256:
`e68297f3d7fb3724fc56fa1632f353774dfc66cadad00528435085eeeef1dec6`.
Failed compile archive SHA256:
`637f4e69cf52a6b350362eda9f42992e43a0e7adace4d9539bc70863064f8d83`.

```text
python3 launch_mpsgraph_prefill.py ROOT ROOT/logs/mpsgraph-prefill-repaired-20261015 checks
python3 archive_mpsgraph_prefill.py ROOT ARCHIVE mpsgraph-prefill-repaired-20261015
python -X utf8 verify_mpsgraph_prefill.py RESULTS VERIFICATION_JSON
```

These measurements establish compatibility and component numerical behavior,
not full-model output, TPS, TTFT or vLLM speedup. The full three-model2x decode
/ half-TTFT goal remains active and unmet.
