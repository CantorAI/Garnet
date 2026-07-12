from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
snapshots = sorted((
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
).glob("*"))
assert snapshots
model_dir = snapshots[-1]
image_path = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
processor = AutoProcessor.from_pretrained(
    str(model_dir), trust_remote_code=True, min_pixels=65536, max_pixels=65536
)
model = AutoModelForImageTextToText.from_pretrained(
    str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
).to("cuda").eval()
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": "Describe the image in one sentence."},
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt"
)
inputs = {name: value.to("cuda") for name, value in inputs.items()}
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=10, do_sample=False)
tail = generated[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
decoded = processor.decode(tail, skip_special_tokens=True)
print(
    "HF Qwen VLM 60-visual-token reference: "
    f"prompt_tokens={inputs['input_ids'].shape[1]}, "
    f"token_ids={tail}, text={decoded!r}"
)
