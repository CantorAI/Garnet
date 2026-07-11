import json
import os
import sys
import time
from pathlib import Path

from common import IMAGE_DIR, env_flag, skip


print("Phase 04: Hugging Face Qwen-VL dataset prompt run")

if not env_flag("RUN_HF_QWEN_VL_DATASET"):
    skip("set RUN_HF_QWEN_VL_DATASET=1 to run one prompt across dataset images")

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor
except Exception as exc:
    skip(f"missing reference dependencies: {exc}")

try:
    from transformers import AutoModelForImageTextToText as AutoQwenVLModel
except Exception:
    try:
        from transformers import AutoModelForVision2Seq as AutoQwenVLModel
    except Exception:
        from transformers import AutoModelForCausalLM as AutoQwenVLModel


MODEL_ID = os.environ.get("HF_QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
PROMPT = os.environ.get(
    "HF_QWEN_VL_PROMPT",
    "Describe the visible people, objects, and scene context in one concise paragraph.",
)
MAX_IMAGES = int(os.environ.get("HF_QWEN_VL_MAX_IMAGES", "3"))
MAX_NEW_TOKENS = int(os.environ.get("HF_QWEN_VL_MAX_NEW_TOKENS", "64"))
MAX_PIXELS = int(os.environ.get("HF_QWEN_VL_MAX_PIXELS", "65536"))
OUTPUT = Path(os.environ.get(
    "HF_QWEN_VL_DATASET_OUTPUT",
    Path(__file__).resolve().parents[2] / "artifacts" / "qwen_vl_reference" / "hf_qwen3_vl_dataset_answers.json",
))

image_list_value = os.environ.get("HF_QWEN_VL_IMAGES", "").strip()
images = (
    [Path(value) for value in image_list_value.split(";") if value.strip()]
    if image_list_value
    else sorted(IMAGE_DIR.glob("*.jpg"))[:MAX_IMAGES]
)
if not images:
    raise FileNotFoundError(f"No JPG files found in {IMAGE_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"model={MODEL_ID}")
print(f"device={device}")
print(f"prompt={PROMPT}")
print(f"max_images={MAX_IMAGES}")
print(f"output={OUTPUT}")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    min_pixels=MAX_PIXELS,
    max_pixels=MAX_PIXELS,
)
model = AutoQwenVLModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
if device == "cpu":
    model = model.to(device)
model.eval()

results = []
for image_path in images:
    started = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = PROMPT

    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_tail = generated_ids[:, input_len:] if input_len else generated_ids
    answer = processor.batch_decode(generated_tail, skip_special_tokens=True)[0].strip()
    if not answer:
        raise AssertionError(f"empty answer for {image_path}")

    item = {
        "image": str(image_path),
        "prompt": PROMPT,
        "answer": answer,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    results.append(item)
    print(f"image={image_path.name}")
    print(answer)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps({
    "model": MODEL_ID,
    "device": device,
    "max_new_tokens": MAX_NEW_TOKENS,
    "results": results,
}, indent=2), encoding="utf-8")

print(f"Wrote {OUTPUT}")
