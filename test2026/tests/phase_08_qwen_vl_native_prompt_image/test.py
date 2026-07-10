import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_native_prompt_image"
PHASE05_SCRIPT = REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "test_real_qwen_mlp_subgraphs.py"
DEFAULT_IMAGE = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
DEFAULT_PROMPT = "Describe this picture in one short sentence."


def extract_decode_json(text):
    marker = '"processor_npz"'
    marker_index = text.find(marker)
    if marker_index < 0:
        marker = '"input_mode"'
        marker_index = text.find(marker)
    if marker_index < 0:
        raise AssertionError("Garnet native prompt-image JSON was not found in subprocess output")
    start = text.rfind("{", 0, marker_index)
    if start < 0:
        raise AssertionError("Garnet native prompt-image JSON start was not found")
    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(text[start:])
    return result


def main():
    image = Path(os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_IMAGE", DEFAULT_IMAGE))
    prompt = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_PROMPT", DEFAULT_PROMPT)
    max_pixels = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_PIXELS", "65536")
    max_new_tokens = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_TOKENS", "10")
    if not image.exists():
        raise AssertionError(f"test image does not exist: {image}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_GARNET_REAL_QWEN_MLP_PARITY": "1",
            "RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_DECODE": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_ONLY": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_IMAGE": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_IMAGE_PREPROCESS": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_JPEG_PREPROCESS": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_TOKENIZER": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_PIXELS": max_pixels,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_TOKENS": max_new_tokens,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROMPT": prompt,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGE": str(image),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-u", str(PHASE05_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_TIMEOUT_SECONDS", "900")),
    )

    combined_output = completed.stdout + "\n" + completed.stderr
    (ARTIFACT_DIR / "last_stdout_stderr.log").write_text(combined_output, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        raise AssertionError(
            f"Garnet native prompt-image subprocess failed with exit code {completed.returncode}. "
            f"See {ARTIFACT_DIR / 'last_stdout_stderr.log'}"
        )

    result = extract_decode_json(combined_output)
    result.update(
        {
            "input_mode": "jpeg_file_and_prompt_to_garnet_tokenizer_nvjpeg_cuda_image_preprocess_then_garnet_model",
            "phase": "phase_08_qwen_vl_native_prompt_image",
        }
    )
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    generated_text = result.get("generated_text", "")
    if not generated_text:
        raise AssertionError("Garnet native prompt-image decode produced empty generated_text")
    print("Phase 08: native prompt+image Garnet generate passed.")
    print(f"image={image}")
    print(f"prompt={prompt}")
    print(f"generated_text={generated_text}")
    print(f"result={ARTIFACT_DIR / 'last_result.json'}")


if __name__ == "__main__":
    main()
