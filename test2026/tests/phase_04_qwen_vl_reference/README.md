# Phase 04: Qwen-VL Reference and Garnet Parity

This phase establishes the first VLM serving correctness harness.

## Goals

1. Use `data/Dataset.1980Love` as a stable image fixture.
2. Run Hugging Face Qwen-VL inference as the reference path.
3. Keep a parallel Garnet test with the same dataset/prompt contract so it can become a parity test as Garnet Qwen-VL support matures.

## Default Behavior

Both heavyweight tests skip with exit code `0` by default:

- `test_hf_qwen_vl.py` skips unless `RUN_HF_QWEN_VL=1`.
- `test_processor_contract.py` skips unless `RUN_HF_QWEN_VL_PROCESSOR=1`.
- `test_weight_contract.py` skips unless `RUN_QWEN_VL_WEIGHT_CONTRACT=1`.
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

Run one prompt across multiple local dataset images:

```powershell
$env:RUN_HF_QWEN_VL_DATASET="1"
$env:HF_QWEN_VL_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
$env:HF_QWEN_VL_PROMPT="Describe the visible people, objects, and scene context in one concise paragraph."
$env:HF_QWEN_VL_MAX_IMAGES="3"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_hf_qwen_vl_dataset.py
```

## Processor Contract Dump

Run when you want exact HF processor tensors for Garnet input parity:

```powershell
$env:RUN_HF_QWEN_VL_PROCESSOR="1"
$env:HF_QWEN_VL_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_processor_contract.py
```

The test writes JSON metadata and NPZ tensors to:

```text
test2026/artifacts/qwen_vl_reference/
```

Generated artifacts are ignored by git.

## Weight Contract

Run when you have a local Qwen-VL config and weights/index:

```powershell
$env:RUN_QWEN_VL_WEIGHT_CONTRACT="1"
$env:GARNET_QWEN_VL_CONFIG_JSON="D:\path\to\config.json"
$env:GARNET_QWEN_VL_WEIGHT_INDEX_OR_DIR="D:\path\to\model.safetensors.index.json"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_weight_contract.py
```

The test expands expected Qwen-VL weight names from config plus the `.x` model files, compares them to local weights, and writes:

```text
test2026/artifacts/qwen_vl_reference/weight_contract_report.json
```

## Garnet Parity Scaffold

Run when Garnet Qwen-VL loading/inference is ready:

```powershell
$env:RUN_GARNET_QWEN_VL="1"
$env:GARNET_DLL_PATH="<build-root>\bin\garnet.dll"
$env:GARNET_QWEN_VL_WEIGHTS="D:\path\to\weights"
.\.venv\Scripts\python.exe test2026\tests\phase_04_qwen_vl_reference\test_garnet_qwen_vl.py
```

The Garnet test currently verifies import, model loading path shape, and forward-call wiring. The temporary input tensors should be replaced with the exact processor/tokenizer tensors from the Hugging Face reference path once Garnet exposes the needed runtime contract.
