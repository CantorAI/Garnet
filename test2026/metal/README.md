# Native Garnet Metal backend bring-up

This investigation uses Garnet's native xlang3 package, captured xModel graphs,
Metal compute kernels and MPS matrix multiplication. It does not currently run
the complete Qwen3.5 model. Complete checkpoint-backed attention, KV state and
vision lowering remain required before text/image inference and model benchmarks.

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
Graph input storage currently accepts contiguous host FP32 rank 1–4 tensors;
checkpoint loading supports F32 and BF16 converted to persistent FP32 matrices.
One command buffer encodes all graph operations, followed by one completion wait
and the final output download. Input allocation, per-context pipeline compilation,
serial RMS reduction and FP32 weight storage still require optimization.

## Executed recurrent component graphs

The existing `Qwen35CausalConvStep` and `Qwen35GatedDeltaStep` xModel programs
now lower their captured state bindings to native Metal kernels. Logical shapes
are preserved while each batch row uses a padded GPU matrix row. Packed state
layouts match the existing backend: convolution groups `[output, history]` per
channel and delta groups `[output, flattened state]` per head. GPU grouped slices
implement the existing xModel unpack helpers.

On the same M4, all 12 captured graph cases passed: convolution and delta,
packed/output/state return paths, batches 1 and 2. Each graph ran three explicit
state updates. The independent NumPy calculations cover inactive-row state
preservation, resuming rows, nonzero initial states and zero Q/K vectors. The
tolerance remains `5e-6 + 5e-5 * abs(reference)`. The final primitive CTest passed
1/1 in 0.06 seconds on the final rerun (0.75 seconds on the initial run).
The existing checkpoint-backed MLP graph regression passed both batches again.
These small components are not full attention layers or
pretrained Qwen inference. The reference inputs include NumPy-computed prior
states; end-to-end decoder feedback is still required.

```sh
/opt/homebrew/bin/python3 source/test2026/metal/test_state_graph.py \
  /Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3 \
  /Users/zsdx/garnet-metal-investigation/build \
  /Users/zsdx/garnet-metal-investigation/source
```

## Complete linear attention and decoder layers

Checkpoint-backed QKV/Z/A/B projections, grouped head expansion, softplus decay,
sigmoid beta, recurrent output RMS normalization/SiLU gating, output projection,
state packing and residual replacement now execute through the captured
`Qwen35LinearAttentionStep` and `Qwen35LinearDecoderLayer` programs. Each graph
retains weights and intermediate matrices and submits one command buffer.

On the M4, all 32 numerical cases produced PASS: batches 1/2, equal/grouped
key/value heads (2/2 and 1/2), attention/decoder graphs and four output modes
(packed, hidden, convolution state, recurrent state). Each case runs seven
updates with mixed BF16/F32 checkpoint values, nonzero initial states and
inactive/resumed batch rows. The tolerance remains `5e-6 + 5e-5 * abs(reference)`.
The fixture uses hidden width 11, key width 7, value width 5 and two value heads;
its timings are component checks, not full model performance.

The initial foreground suite lost its SSH connection after completing batch 1
and the first batch-2 case. Process inspection then confirmed it was no longer
running. All batch-2 cases were rerun with durable logs and completed with exit
code 0 under `logs/linear-batch2-20261011` on the Mac. The native CTest also passed
1/1 (3.96 seconds including startup and shader compilation).

```sh
/opt/homebrew/bin/python3 source/test2026/metal/test_linear_graph.py \
  /Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3 \
  /Users/zsdx/garnet-metal-investigation/build \
  /Users/zsdx/garnet-metal-investigation/source
```

`background_test.py` can preserve test output and exit evidence across SSH drops.
Its download mode reuses the existing pinned-checkpoint downloader. Direct model
download failed on this Mac with a DNS/routing error; no pretrained inference is
claimed. The existing full checkpoint was found on the Pi (1,746,942,600 bytes),
and transfer through the local investigation workspace is underway.

Next: verify full-size pretrained linear layers, implement full attention/RoPE/KV
updates and vision encoding, then run actual text/image inference and profile
latency, memory and throughput before optimizing the complete workload.

## Full attention implementation

Captured full-attention and decoder programs now lower Q/K/V and output
projections, per-head RMS normalization, partial/interleaved THW RoPE, explicit
KV updates, grouped-query attention, stable masked softmax and sigmoid output
gating to Metal/MPS. Cache unpack and decoder hidden replacement reuse the
GPU grouped-slice/concat operations. Active slots and finite RoPE positions are
validated before GPU dispatch; an inactive row preserves both KV caches.
Allocations exceeding the device's maximum buffer length are rejected.

The new independent NumPy test exercises 32 cases: attention/decoder, batches
1/2, partial/full RoPE, and packed/hidden/key/value output modes, each with seven
steps and inactive/resumed rows. Invalid active cache slots must also be rejected.
All 32 cases passed on the actual M4, and the durable suite exit code was 0.
The actual native CTest passed 1/1 in 3.28 seconds after fixing a Metal type-name
collision. The full suite's durable logs are under `logs/attention-20261011` on
the Mac. No full-size pretrained result is established by these fixtures.

```sh
/opt/homebrew/bin/python3 source/test2026/metal/test_attention_graph.py \
  /Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3 \
  /Users/zsdx/garnet-metal-investigation/build \
  /Users/zsdx/garnet-metal-investigation/source
```

The Pi-to-PC checkpoint copy completed: 1,746,942,600 bytes, SHA256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The resumable upload to the isolated Mac model folder remains incomplete.
Mac DNS queries failed and a direct IPv4 HTTPS probe reset; system networking
settings were not changed. `test_pretrained_graph.py` prepares complete layer-0
linear and layer-3 full decoder tests from the official checkpoint using mapped
weight reads, four steps, batches 1/2 and child peak RSS reporting.

Full text generation, vision encoding, actual image inference and workload
performance measurements/optimization remain required.
