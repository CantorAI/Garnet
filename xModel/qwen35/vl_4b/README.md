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
