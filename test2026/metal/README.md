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

## Native decoder serving adapter

The Metal package now exports the existing Qwen tokenizer plus host tensor I/O
for BF16 embedding lookup, zero state creation, packed state extraction and
argmax. Learned decoder operations and the vocabulary projection execute on
Metal/MPS. `tensor_io.logits` accepts already projected vocabulary rows; it
rejects unprojected hidden rows instead of performing a CPU projection.
`MetalDecoder` reuses the explicit-state serving interface and captures layer
graphs with explicit tensor parameters. Chunked prefill is not implemented.

Actual Mac M4 closed-loop tests passed for batches 1 and 2: eight predictive
steps, independent NumPy token/state references, and an inactive/resumed row.
The durable job `logs/model-explicit-capture-20261011` exited 0. Last-step
layer/head times were 1.100/0.400 ms for batch 1 and 1.319/0.685 ms for batch 2.
These are small synthetic single-layer fixture timings, not Qwen performance.
The first two adapter attempts failed capture; explicit fusion parameters
resolved the failure, without changing tolerances or using a CPU neural path.

Backend identification now queries the device without compiling shaders.
Command encoding/allocation/submission uses autorelease pools to bound
temporary Objective-C objects in a pure C++ caller. No full-model memory
improvement is claimed before measurement.

`benchmark_text.py` and `run_text_decoder.py` prepare actual native generation,
two requests through a loaded model, bounded output, TTFT, per-step/layer/head
timings and macOS peak runtime RSS. They remain unexecuted until the complete
pinned checkpoint is available on the Mac. The transfer resumes its partial
file over the verified ZeroTier route; no system networking changes are needed.

## Native still-image encoder

The `qwen35_vision_encoder` capture now lowers the complete fixed-grid still
image tower: flattened 3D patch projection and bias, bilinear position-table
interpolation, mean/variance layer normalization, vision RoPE, noncausal
multihead softmax attention, residual projections, tanh GELU MLPs, spatial
merging and erf GELU output projection. Learned operators run on Metal/MPS;
host preprocessing prepares RGB patches and constant coordinates. MPS row
padding is respected when flattening and merging token rows. Batch 1/2 and
equal grids per batch row are supported. This remains synchronous FP32 code.

Actual M4 tests of `MetalImageEncoder` passed four batch/grid combinations:
batches 1/2 with grids 2x2 and 4x6, each evaluated on two inputs. The fixture
has two vision blocks, hidden width 16 and BF16 checkpoint weights. The durable
`logs/vision-serving-20261011` job exited 0; maximum absolute error across
these cases was 5.37e-7, below the unchanged `5e-6 + 5e-5 * abs(reference)`
tolerance. The native CTest also passed 1/1 in 0.85 seconds after removal of
temporary diagnostic output selectors. Fixture timings are not model timings.
The final file-input suite `logs/vision-file-serving-20261011` also exited 0:
all eight evaluations used native FP32 file reads and validated per-batch
visual-token extraction. Its maximum absolute error was 5.96e-7. Both attention
contractions in this suite use explicit einsum reference calculations.
The decoder closed-loop batch-1/2 regression job also exited 0 after this
addition (`logs/model-after-vision-20261011`, eight steps per batch).

The first comparison exposed a reference issue: on this Mac's NumPy 2.3.4,
strided `prob @ value` disagreed with explicit einsum by 1.528. The identical
Windows NumPy 2.3.5 fixture agreed within 5.96e-8, and its attention context
matched GPU output. `reference_metal_vision.py` uses explicit einsum attention
contractions; model equations and tolerances were retained. The original Pi
reference was left in place.

`test_pretrained_vision_graph.py` prepares an independent comparison of all
12 official vision blocks and the merger. `run_image_decoder.py` and
`benchmark_image.py` prepare two actual image-plus-text requests with both
models resident, fresh image encoding and decoder reset for each request,
bounded output, vision/prefill/TTFT/decode timings and peak runtime RSS. They
have not executed against pretrained weights yet.

The real test image and prepared FP32 patches are on the Mac under `images/`.
Preprocessing resizes 1818x1372 to 288x192, yielding grid 18x12, 216 patches
and 54 visual tokens. Mac file hashes match the local originals:

- Image: `eba5d9b6124fbd3b2a809e1c664886c29d842406134bb4c1b281414d0c7101b9`.
- Batch-1 patches: `1f82fddd12eeff26e32727d0d92a742fda8259cfbecbcb171d568df12a752b15`.
- Batch-2 patches: `bd417b80148a5c1d871beae7d53a680b57ba8a11c1004f64f199a220b215b1a5`.

`transfer_checkpoint.py` now checks each complete 4 MiB block before append,
verifies an existing prefix before resuming after a lost acknowledgement, and
requires the full pinned SHA256 before promoting the partial checkpoint.
Actual verified appends have progressed beyond 152 MB. The transfer is still
running; pretrained text/image generation, full-size correctness checks,
workload memory/performance measurements and optimization remain outstanding.

## Whole serving-path fixture and direct checkpoint source

The combined vision-to-decoder test passed on the actual M4 for batches 1/2
(`logs/image-pipeline-20261011`, exit 0). Each batch evaluates two different
images through two vision blocks, a linear decoder block, a full-attention
decoder block and the tied GPU vocabulary head. Both requests run 14 steps;
predictions and all recurrent/KV states match independent NumPy references
within the original tolerance. The second request resets state while retaining
weights; batch 2 also pauses/resumes a row. This is a small BF16 integration
fixture, not a pretrained caption or a full-model benchmark.

The SSH transfer gained lossless block compression and durable JSONL logs
under local `work/logs/checkpoint-transfer.jsonl`. A representative 16 MiB
block compressed to 13,441,235 bytes in 0.684 seconds. Compressed 4 MiB
blocks were received/verified on the Mac. The uploader was deliberately
stopped at a verified prefix after a faster direct source was found.

The official ModelScope file listing advertises the exact pinned checkpoint
size and SHA256. A direct Mac 1 MiB range probe succeeded at 1,474,454 bytes/s;
its SHA256 `667d46579c60b536482da00c4cef612400f9f51af95f93f513021e2e6109c88a`
matches the same prefix of the locally verified checkpoint. The background
job `logs/modelscope-download-20261011` uses `download_modelscope_checkpoint.py`
to resume the existing partial file, verify the full SHA256 before promotion,
and verify config/tokenizer/preprocessor files individually. No system
network settings were changed. Pretrained validation and benchmarks follow
only after that job finishes successfully.
