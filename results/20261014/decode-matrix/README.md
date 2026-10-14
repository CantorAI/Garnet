# FP32 matrix units at decode rows: slower than native dot kernels

This component tests the existing FP32 SIMD-group matrix kernel at actual
decode batch sizes 1 and 2, plus an eight-row internal tile derived from it.
Unused internal rows contain zero; request batch size and values are unchanged.
Weights retain canonical BF16 bits and arithmetic remains FP32. The control
retains the native small-output kernel for columns <=256 and regular decode
kernel otherwise. Prefill already uses matrix units; this experiment concerns
padding matrix computation for decode.

Owned PID 31524 exited 0 after 44.27 seconds. All 34 matrix fixtures (25 trained
and nine synthetic tails), both batches and both candidates passed 136
numerical/padding checks under the existing FP64 envelope. Worst normalized
FP64 error was 0.0460314187, below 1. Reduction order differs, so this is not a
bitwise identity claim. The evidence contains 1,088 alternating timing windows.

| Candidate | Pooled candidate/control GPU duration |
| --- | ---: |
| Eight-row internal matrix tile | 1.80993 |
| Original sixteen-row matrix tile | 2.47925 |

Both are substantially slower. No full-model integration or speed claim is
warranted. The selected runtime libraries remain unchanged; the full 2x decode
and half first-token latency goal remains unmet. Component timings do not
establish end-to-end TPS or first-token latency.

```text
python3 launch_decode_matrix_probe.py ROOT ROOT/logs/decode-matrix-probe-20261014
python3 archive_decode_matrix_probe.py ROOT ARCHIVE
python verify_decode_matrix_probe.py RESULTS_DIRECTORY VERIFICATION_JSON
```

Archive SHA-256:
`c95720f52d0f64e8034899fc8de166b409c3c345ff214b6024ad8d2e77fb084b`.
The local verifier checks 41 archived file hashes, source identities, terminal
stages, numerical gates and independently recomputes GPU timing ratios. Large
canonical matrix bytes are omitted with 34 checkpoint/slice hash proofs.
Executed sources are frozen in the archive. The runner's inherited description
mentions sub-SIMD, but the generated shader and recorded candidate names identify
the matrix experiment. OS caches and GPU frequency are uncontrolled, and the
0.5-second process guard cannot exclude arbitrary shorter external activity.
