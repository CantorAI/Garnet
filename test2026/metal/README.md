# Native Garnet Metal backend bring-up

This investigation uses Garnet's native xlang3 package, captured xModel graphs,
Metal compute kernels and MPS matrix multiplication. It does not currently run
the complete Qwen3.5 model. Attention, recurrent/KV state and vision lowering
remain required before text/image inference and meaningful model benchmarks.

## Verified on Apple M4, 16 GiB, 2026-10-11

The existing Mac checkout is `/Users/zsdx/CantorAI/Garnet`, clean `main` at
`f133306`. The SDK and runtime are reused from the existing CantorAI build.
The investigation source and build are isolated under
`/Users/zsdx/garnet-metal-investigation`; shared checkouts/builds are read-only.

The native library compiled with AppleClang 17.0.0. The numerical Metal/MPS
CTest passed 1/1 in 0.74 seconds including process startup. It covers padded
matrix strides, transposed weights, chained operations, repeated requests,
RMS normalization, SiLU gating and pending/aborted output protection.

The checkpoint-backed integration test captures the actual xModel normalization,
MLP and residual helpers, masks inactive rows, and applies the tied output head.
Weights are independently generated and BF16-rounded; NumPy computes the
reference from those checkpoint values. Each captured graph runs twice with
persistent weights and GPU intermediates. The expanded test passed:

| Batch | Compile ms | Last run ms |
| --- | ---: | ---: |
| 1 | 94.945083 | 1.102125 |
| 2 | 35.450667 | 1.056834 |

These are tiny synthetic component fixtures (hidden width 12, MLP width 20,
vocabulary 19), not complete model latency, tokens/sec or memory measurements.
The earlier norm/MLP/head-only fixture also passed both batches.

## Reproduce on the Mac

```sh
cd /Users/zsdx/garnet-metal-investigation
/opt/homebrew/bin/cmake -S source -B build \
  -DGARNET_METAL=ON -DGARNET_METAL_XMODEL=ON \
  -DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk \
  -DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib \
  -DCMAKE_BUILD_TYPE=Release
/opt/homebrew/bin/cmake --build build -j4
/opt/homebrew/bin/ctest --test-dir build --output-on-failure
/opt/homebrew/bin/python3 source/test2026/metal/test_model_graph.py \
  /Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3 \
  /Users/zsdx/garnet-metal-investigation/build \
  /Users/zsdx/garnet-metal-investigation/source
```

The Python reference requires NumPy (existing Mac installation: 2.3.4).
Graph input storage currently accepts contiguous host FP32 vectors/matrices;
checkpoint loading supports F32 and BF16 converted to persistent FP32 matrices.
One command buffer encodes all graph operations, followed by one completion wait
and the final output download. Input allocation, per-context pipeline compilation,
serial RMS reduction and FP32 weight storage still require optimization.
