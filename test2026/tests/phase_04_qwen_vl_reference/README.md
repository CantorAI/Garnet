# Phase 04: Qwen-VL Reference and Garnet Parity

This phase establishes the first VLM serving correctness harness.

## Goals

1. Use `data/Dataset.1980Love` as a stable image fixture.
2. Run Hugging Face Qwen-VL inference as the reference path.
3. Keep a parallel Garnet test with the same dataset/prompt contract so it can become a parity test as Garnet Qwen-VL support matures.

## Default Behavior

Both heavyweight tests skip with exit code `0` by default:

- `test_hf_qwen_vl.py` skips unless `RUN_HF_QWEN_VL=1`.
- `test_garnet_qwen_vl.py` skips unless `RUN_GARNET_QWEN_VL=1`.

This keeps `test2026/run_tests.py` fast and avoids accidental model downloads or hard failures while Garnet runtime support is still under development.

## Hugging Face Reference

Run:

```powershell
$env:RUN_HF_QWEN_VL="1"
$env:HF_QWEN_VL_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_hf_qwen_vl.py
```

Optional:

- `HF_QWEN_VL_MAX_NEW_TOKENS`
- `HF_QWEN_VL_OUTPUT`

## Garnet Parity Scaffold

Run when Garnet Qwen-VL loading/inference is ready:

```powershell
$env:RUN_GARNET_QWEN_VL="1"
$env:GARNET_DLL_PATH="D:\CantorAI2026\Garnet\out\build\x64-Debug\bin\garnet.dll"
$env:GARNET_QWEN_VL_WEIGHTS="D:\path\to\weights"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_garnet_qwen_vl.py
```

The Garnet test currently verifies import, model loading path shape, and forward-call wiring. The temporary input tensors should be replaced with the exact processor/tokenizer tensors from the Hugging Face reference path once Garnet exposes the needed runtime contract.
