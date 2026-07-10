# Qwen-VL Native Image Pipeline

This phase removes the HF image processor from the Garnet model input path and
adds an NVIDIA-native JPEG ingest path inspired by xWorld's `HwImageProcessor`:
JPEG bytes decode to GPU RGB8 with nvJPEG, then CUDA kernels produce Qwen-VL
patch-layout tensors.

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

The C ABI also exposes NVIDIA-owned image ingest:

- `GarnetQwenVLPreprocessJpegFile(...)`
- `GarnetQwenVLPreprocessJpegBytes(...)`

Those APIs accept JPEG file/bytes, decode with nvJPEG into GPU RGB8, smart-resize
to Qwen-VL dimensions, and emit:

- `pixel_values`: Qwen-VL flattened patch tensor on host for the current Python bridge
- `image_grid_thw`: `[1, grid_h, grid_w]`
- metadata: source height/width and resized height/width

The raw RGB path remains intentionally supported because an upper pipeline may
already own decode/crop/ROI and can pass raw frames directly to Garnet.

## Backend Slots

- `cuda_raw_tensor`: implemented
- `cuda_jpeg_nvjpeg`: implemented for JPEG file/bytes -> GPU RGB8
- `cuda_resize_patch_layout`: implemented for RGB8 -> Qwen-VL patch layout
- `cuda_resize_npp_or_custom`: implemented for raw float RGB; exact HF resize parity is not required yet
- `qwen_tokenizer_bpe`: implemented for Qwen3-VL byte-level BPE prompt/decode path
- `qwen_vl_prompt_builder`: implemented for single-image chat prompt and image-pad insertion
- `cuda_roi_crop`: planned
- `cuda_nv12_import`: planned
- `preprocessed_pixel_values`: existing debug/bridge path

## Tests

- Phase 07 compares Garnet CUDA patch layout against HF processor output.
- Phase 08 feeds Garnet tokenizer IDs and nvJPEG/CUDA-produced `pixel_values` into the existing Qwen3-VL model decode path.
- Phase 09 validates original raw RGB -> GPU resize -> Qwen patch layout.
- Phase 10 validates JPEG file -> nvJPEG GPU decode -> CUDA resize/patch layout against HF processor grid and mean drift.
- Phase 11 validates native Qwen tokenizer special IDs, prompt IDs, and decode against HF tokenizer.

Latest local checkpoint:

- `frame_0.jpg` source: `1080x1920`
- resized Qwen input: `192x320`
- `image_grid_thw`: `[1, 12, 20]`
- `pixel_values`: `[240, 1536]`
- Qwen-VL prompt tokens: `79`
- phase 10 mean absolute drift vs HF processor: `0.0384`
- phase 08 generated text from Garnet path: `A man sits in a rustic wooden hut, surrounded`

HF remains a test oracle for tokenizer/image parity, not the runtime tokenizer or image preprocessor for phase 08.
