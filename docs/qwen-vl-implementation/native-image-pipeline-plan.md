# Qwen-VL Native Image Pipeline

This phase removes the HF image processor from the Garnet model input path.

## Current Implemented Contract

`garnet.qwen_vl_preprocess_image(...)` accepts an upstream-provided raw image tensor:

- layout: float32 HWC
- formats: RGB, BGR, RGBA, BGRA
- value range: 0..255 by default
- dimensions: already resized to Qwen-VL smart-resize dimensions

It returns:

- `pixel_values`: Qwen-VL flattened patch tensor
- `image_grid_thw`: `[1, grid_h, grid_w]`
- metadata: backend, patch size, merge size, resized height/width

This is intentionally GPU-first. Garnet does not own CPU decode. A caller can provide JPEG bytes to a future nvJPEG source, or raw CPU/GPU image memory from an upstream pipeline.

## Backend Slots

- `cuda_raw_tensor`: implemented
- `cuda_jpeg_nvjpeg`: planned
- `cuda_resize_npp_or_custom`: planned
- `cuda_roi_crop`: planned
- `cuda_nv12_import`: planned
- `preprocessed_pixel_values`: existing debug/bridge path

## Tests

- Phase 07 compares Garnet CUDA patch layout against HF processor output.
- Phase 08 feeds Garnet-produced `pixel_values` into the existing Qwen3-VL model decode path.

HF remains a test oracle and tokenizer reference, not the runtime image preprocessor for phase 08.
