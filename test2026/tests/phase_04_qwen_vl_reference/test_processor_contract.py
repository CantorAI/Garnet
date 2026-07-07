import json
import os
import time
from pathlib import Path

from common import REPO_ROOT, env_flag, first_dataset_sample, skip


print("Phase 04: Hugging Face Qwen-VL processor contract dump")

if not env_flag("RUN_HF_QWEN_VL_PROCESSOR"):
    skip("set RUN_HF_QWEN_VL_PROCESSOR=1 to dump HF processor tensors")

try:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
except Exception as exc:
    skip(f"missing processor dump dependencies: {exc}")


PROMPTS = {
    "objects_json": """Detect visible objects in this image. Return JSON only:
{
  "objects": [
    {
      "type": "person|vehicle|bicycle|sign|animal|object|other",
      "bbox": [x1, y1, x2, y2],
      "confidence": "low|medium|high",
      "description": "short phrase"
    }
  ]
}""",
    "scene_summary": "Describe the scene in one short sentence, focusing on people, vehicles, objects, and activity.",
    "event_verify": """Return JSON only:
{
  "event": "entering|leaving|waiting|interacting|moving|unknown",
  "evidence": "one short sentence"
}""",
    "safety_check": """Return JSON only:
{
  "hazards": [
    {
      "type": "traffic|crowd|blocked_path|fall_risk|unknown",
      "description": "short phrase"
    }
  ]
}""",
}

MODEL_ID = os.environ.get("HF_QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
ARTIFACT_DIR = Path(os.environ.get(
    "HF_QWEN_VL_PROCESSOR_OUTPUT_DIR",
    REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_reference",
))


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def array_summary(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def build_chat_text(processor, image, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


image_path, _metadata = first_dataset_sample()
image = Image.open(image_path).convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"model={MODEL_ID}")
print(f"processor={processor.__class__.__name__}")
print(f"image={image_path}")
print(f"artifact_dir={ARTIFACT_DIR}")

written = []
for prompt_id, prompt in PROMPTS.items():
    started = time.perf_counter()
    chat_template = build_chat_text(processor, image, prompt)
    inputs = processor(text=[chat_template], images=[image], return_tensors="pt")

    arrays = {}
    for key, value in inputs.items():
        array = to_numpy(value)
        arrays[key] = array

    if "input_ids" not in arrays:
        raise AssertionError(f"processor output missing input_ids for prompt_id={prompt_id}")
    if "attention_mask" in arrays and arrays["attention_mask"].shape != arrays["input_ids"].shape:
        raise AssertionError(
            f"attention_mask shape {arrays['attention_mask'].shape} != input_ids shape {arrays['input_ids'].shape}"
        )

    stem = f"processor_{image_path.stem}_{prompt_id}"
    npz_path = ARTIFACT_DIR / f"{stem}.npz"
    json_path = ARTIFACT_DIR / f"{stem}.json"
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "schema_version": "0.1",
        "model": MODEL_ID,
        "processor_class": processor.__class__.__name__,
        "image": str(image_path),
        "prompt_id": prompt_id,
        "prompt": prompt,
        "chat_template": chat_template,
        "arrays": {key: array_summary(value) for key, value in arrays.items()},
        "npz": str(npz_path),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reloaded = np.load(npz_path)
    for key, original in arrays.items():
        if key not in reloaded:
            raise AssertionError(f"reloaded NPZ missing key={key} prompt_id={prompt_id}")
        if reloaded[key].shape != original.shape:
            raise AssertionError(
                f"reloaded shape mismatch key={key}: {reloaded[key].shape} != {original.shape}"
            )
        if str(reloaded[key].dtype) != str(original.dtype):
            raise AssertionError(
                f"reloaded dtype mismatch key={key}: {reloaded[key].dtype} != {original.dtype}"
            )

    if prompt not in chat_template:
        raise AssertionError(f"prompt text missing from chat template for prompt_id={prompt_id}")

    print(f"prompt_id={prompt_id}")
    for key, value in arrays.items():
        print(f"  {key}: shape={list(value.shape)} dtype={value.dtype}")
    print(f"  wrote_json={json_path}")
    print(f"  wrote_npz={npz_path}")
    written.append((json_path, npz_path))

if not written:
    raise AssertionError("processor contract wrote no artifacts")

print(f"Processor contract wrote {len(written)} prompt dumps.")
