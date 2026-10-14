# Dedicated MPS vector multiply: dtype limit and rejected FP32 route

Actual execution on the M4 establishes two results. A Float16 matrix with
FP32 input/output vectors aborts with the framework assertion `Both inputs
must use the same type.` The valid all-FP32 route passes real-weight checks
but is much slower than the compact-BF16 native control. Neither route is
integrated into serving. The full 2x decode/half-TTFT goal remains unmet.

## Actual compatibility

The dedicated `MPSMatrixVectorMultiplication` API differs from the previously
tested mixed-input `MPSMatrixMultiplication`. Each dtype combination executes
in its own guarded child process using exact dyadic values, five outputs and
37 input elements. The unsupported combination terminates only its child
with SIGABRT (exit -6); the following all-FP32 child exits 0 and matches all
five independent expected outputs exactly. The parent finishes with a status
record for both outcomes; its exit 0 does not mean both combinations passed.

This rules out directly substituting the existing exact-half weight cache
while retaining FP32 activation vectors through this API on the tested Mac.
No activation conversion, rounding or precision shortcut was introduced.
Apple's [matrix-vector API documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixvectormultiplication?language=objc)
identifies the operation; the mixed-dtype limit above is measured framework
behavior on this system, not a claim about every device or future OS.

## Real-matrix component measurements

The valid route losslessly expands original BF16 bits into FP32 matrix storage
with a bit shift, preserving every learned value. Input, result and matrix
types are FP32. Batch 2 encodes one vector operation per row into a single
command buffer. The control shares compact BF16 weights across both rows.
Weight widths and memory traffic therefore differ; this is not a comparison
with identical storage, nor a stock-runtime performance claim.

| Model | Batch | Pooled large-matrix control/candidate GPU ratio | Body-only ratio |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 0.5141 | 0.5724 |
| Qwen3-0.6B | 2 | 0.2537 | 0.3365 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.4999 | 0.5557 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.2579 | 0.3119 |
| Qwen3.5-0.8B | 1 | 0.5067 | 0.7026 |
| Qwen3.5-0.8B | 2 | 0.2545 | 0.3395 |

Ratios below one mean slower. Pools sum measured GPU times once per collected
matrix/batch with more than 256 outputs; body-only also excludes vocabulary
heads. They are not weighted by model layer counts and are not decode TPS.

| Model | Batch | Control full-head GPU ms | FP32 GEMV full-head GPU ms |
| --- | ---: | ---: | ---: |
| Text | 1 | 3.166 | 6.276 |
| Text | 2 | 2.996 | 12.336 |
| Coder | 1 | 2.719 | 5.529 |
| Coder | 2 | 2.670 | 10.666 |
| VL | 1 | 4.993 | 10.243 |
| VL | 2 | 5.092 | 20.570 |

Every real-matrix profile passes the unchanged `5e-6+5e-5*abs(control)`
comparison and the independent FP64 envelope
`gamma(ceil(K/32)+32)*sum(abs(products))+1e-7`. Both control and candidate are
checked against FP64; worst candidate allowed-bound ratio is 0.0097090214.
All 68 matrix/batch checks pass, including first-layer projections, complete
heads, irregular tails, distinct batch inputs and output-padding preservation.
Output summation order may differ; general bit-identical output is not claimed.

Each profile has two ABBA rounds, three warm operations per window and twelve
measured operations per window. Preparation, FP32 expansion, object creation,
source compilation and numerical readbacks are outside component timing.
GPU and host wall times are retained. Wider storage and separate per-row work
may contribute to the slowdown, but DRAM traffic/cause is not independently
measured. Resident startup and peak process memory are not measured here.

This large regression rejects integration without a favorable-number repeat
or costly full-model variant. No complete generation or new vLLM comparison
was run; serving sources and primary binaries remain unchanged.

## Evidence and reproduction

Owned root: `/Users/zsdx/garnet-metal-investigation`. Compatibility PID 53831
and component PID 54260 are terminal. Conflict guards reject competing
inference/probes and enforce at least 2 GiB free space. No unrelated process
was stopped and no comparator dependencies were installed.

Commands are `run_mps_gemv_compat.py ROOT NEW_OUTPUT_DIR` and
`run_mps_gemv_probe.py ROOT NEW_OUTPUT_DIR`. Exact absolute compiler/runtime
commands, source snapshots/hashes, raw assertion, statuses, hardware, model
and matrix identities, numerical/timing windows and monitors are in
[evidence](evidence/). Raw matrices remain private and are omitted with 34
verified size/hash/checkpoint proofs; they are regenerable from the pinned
checkpoints. The source includes the lossless expansion and vector-view setup.

```text
python test2026/metal/verify_mps_gemv.py results/20261013/mps-gemv
```

The verifier checks 62 archive file hashes, actual -6/0 dtype outcomes,
source/checkpoint identities, all 68 numerical/padding checks and timing
arithmetic. Archive SHA256:
`1b06838f8dcd77658580c16619ff59a4b36af6cc5e38a319af3921ca19a75a07`.
The primary/attention libraries and GPT-OSS/Pi checkouts remain intact.
The [latest actual vLLM comparison](../selected-attention-vllm/README.md)
continues to record all six profiles and their unmet joint targets.
