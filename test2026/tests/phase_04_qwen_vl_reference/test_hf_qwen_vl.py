import json
import os
from pathlib import Path

from common import env_flag, first_dataset_sample, qwen_prompt, skip


print("Phase 04: Hugging Face Qwen-VL reference inference")

if not env_flag("RUN_HF_QWEN_VL"):
    skip("set RUN_HF_QWEN_VL=1 to run transformer inference; normal CI avoids model downloads")

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


model_id = os.environ.get("HF_QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
max_new_tokens = int(os.environ.get("HF_QWEN_VL_MAX_NEW_TOKENS", "32"))
output_path = Path(os.environ.get("HF_QWEN_VL_OUTPUT", Path(__file__).with_name("hf_reference_output.json")))

image_path, metadata = first_dataset_sample()
prompt = qwen_prompt(metadata)
image = Image.open(image_path).convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"model={model_id}")
print(f"device={device}")
print(f"image={image_path}")
print(f"prompt={prompt}")

try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoQwenVLModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_tail = generated_ids[:, input_len:] if input_len else generated_ids
    answer = processor.batch_decode(generated_tail, skip_special_tokens=True)[0].strip()

    output = {
        "model": model_id,
        "image": str(image_path),
        "prompt": prompt,
        "answer": answer,
        "device": device,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    assert answer, "Qwen-VL reference generated an empty answer"
    print(answer)
    print(f"Wrote {output_path}")
except Exception as exc:
    print(f"HF Qwen-VL reference inference failed: {exc}")
    raise
