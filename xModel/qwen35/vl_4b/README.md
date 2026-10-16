# Qwen3.5 4B Metal xModel

Use `xModel/qwen35/vl_4b` as the Galaxy `xmodelRoot` with the official
Qwen/Qwen3.5-4B checkpoint, including both safetensors shards and their index.
The model contract validates the 32-layer decoder, 32 linear value heads,
2560-wide hidden state, and 24-layer vision encoder before graph allocation.
It reuses the config-driven Qwen3.5 graph engine; it does not copy the 0.8B
architecture constants or use a different inference framework.

Serving profiles expose `prefillChunk` (default 16), `imagePixels` (default
65536), and the existing input/output capacity. Increasing resolution or
prefill width needs an accuracy, memory, and latency measurement on the target
machine. A profile is not an accuracy guarantee.

The checked-in `serving-profile.json` uses 16-token parallel linear prefill.
The 4B packed hidden width exceeds the current admitted limit at chunk 32;
serving rejects that profile before loading decoder graphs. The shared engine
supports both the 4B grouped linear attention (16 key / 32 value heads) and its
32-layer decoder.

Reproduce the serving comparison using the same fixtures in two separate
processes (loading two 4B instances together would distort memory results):

```sh
xlang3 -B xModel/qwen35/vl_4b/benchmark.py MODEL_ROOT serial BUNNY_JPEG CAR_JPEG STREET_JPEG
xlang3 -B xModel/qwen35/vl_4b/benchmark.py MODEL_ROOT parallel BUNNY_JPEG CAR_JPEG STREET_JPEG
```

Set `GARNET_METAL_FRONTEND_PYTHON` to the NumPy/Pillow frontend environment and
run with the built native Garnet module available. The helper does image
preparation only; inference uses Garnet Metal. Record `/usr/bin/time -l` output
separately to compare process footprint. The original fixture hashes and actual
MacBook comparison are documented in the WorldSense physical AI experiment report.

## Optional native kernel profile

The serving JSON's `linearChunk` selects the graph algorithm; it does not enable
the optional native chunk kernels. For a separate Metal build, pass
`-C cmake/MetalWorldSense4B.cmake` at first configuration, together with the
existing Metal/xModel, SDK and runtime settings. This opts into six low-copy
kernel policies without changing checkpoint precision, image pixels or outputs.
It leaves global defaults and CUDA builds unchanged. Explicit `-D...=OFF`
overrides remain available; inspect CMakeCache.txt rather than assuming flags.

Two fresh M5/24 GiB control/candidate cohorts retained identical full generated
token sequences on three fixture images. Warm median total time improved about
23–25%, with roughly 48% lower TTFT. These are a small serving regression sample,
not a full numerical validation, general speed guarantee or M4/vLLM comparison.
See `results/20261016/4b-m5-chunk/README.md` for individual measurements, memory
costs, source/binary identities and remaining validation.

## Mixed-video shapes and further decode tuning

The optional profile now also enables BODY_SHAPE (body projections only), and
4B Metal serving defaults to `shareVisionWeights: true`. Fixed-grid replacements
share checked immutable native weight buffers and release the prior graph only
after the replacement is ready. Fresh images are always encoded; no image or
answer cache is added. Explicit false disables sharing.

Two fresh cohorts against the prior six-flag optimized build measured another
17–20% lower median total time and38–43% lower TTFT, with7–8% higher decode rate
on the three-image smoke workload. Complete tokens, sampled32-layer state bits,
vision embedding bits and source-close/failure recovery checks passed. The
sharing-only attempt failed a decode-preservation gate and is retained as
rejected evidence. See results/20261016/4b-m5-vision-shape/README.md.
For an existing build cache, explicitly set GARNET_METAL_BODY_SHAPE=ON; a cache
initializer does not overwrite an existing OFF value.
