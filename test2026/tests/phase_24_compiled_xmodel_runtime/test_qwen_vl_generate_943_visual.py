# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GARNET_DLL = REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin" / "garnet.dll"
handles = []
for directory in [
    GARNET_DLL.parent,
    REPO_ROOT.parent / "out" / "build" / "x64-Release" / "bin",
    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
]:
    if directory.exists() and hasattr(os, "add_dll_directory"):
        handles.append(os.add_dll_directory(str(directory)))

import xlang3


snapshots = sorted((
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
).glob("*"))
assert snapshots
image_path = Path(os.environ.get(
    "GARNET_BALANCED_IMAGE_PATH",
    REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg",
))
assert image_path.exists()

patch_count = 3772
visual_token_count = 943
max_tokens = 1024
kv_pages = 80
garnet = xlang3.importModule("garnet", fromPath=str(GARNET_DLL))
load_start = time.perf_counter()
model = garnet.load_model(
    str(
        REPO_ROOT
        / "xModel"
        / "qwen3"
        / "vl_2b_instruct"
        / "qwen_vl_prefill.py"
    ),
    runtime_mode="compiled_xmodel",
    entry_function="Qwen3VLPrefill",
    frontend="qwen3_vl",
    weights=str(snapshots[-1]),
    input_shapes=[
        [1, max_tokens], [patch_count, 1536], [1, 3],
        [patch_count, 4], [patch_count, 4], [patch_count, 2], [2],
        [1, max_tokens], [1, max_tokens], [3, 1, max_tokens], [1, 1],
        [28, kv_pages, 16, 8, 128], [28, kv_pages, 16, 8, 128],
        [kv_pages], [1],
    ],
    input_dtypes=[
        "int64", "bfloat16", "int64", "int64", "bfloat16", "int64", "int32",
        "int64", "int64", "int64", "int64", "bfloat16", "bfloat16", "int32", "int32",
    ],
    compile={"builder_workspace_mb": 4096},
    cache_dir=os.environ.get(
        "GARNET_BALANCED_CACHE_DIR",
        str(SCRIPT_DIR / "qwen_vl_generate_943_visual_cache"),
    ),
)
status = model.runtime_status()
assert bool(status["ready"]), status
assert bool(status["engines_prepared"]), status
assert bool(status["frontend_prepared"]), status
assert int(status["partition_options"]["builder_workspace_bytes"]) == 4096 << 20, status
plan = json.loads(status["execution_plan_json"])
assert status["engine_partition_count"] >= 2, status
assert any(
    operation["partition_reason"].startswith("preferred:")
    for operation in plan["operations"]
), plan
load_ms = (time.perf_counter() - load_start) * 1000.0

request = {
    "image_path": str(image_path),
    "prompt": os.environ.get(
        "GARNET_BALANCED_PROMPT",
        (
            'Detect the most prominent person. JSON only: {"description": string under 6 words, '
            '"persons": [{"bbox": [x1,y1,x2,y2]}]}. Coordinates are integers 0-1000. '
            "Return one box, or [] if no person. No markdown."
        ),
    ),
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1280 * 28 * 28,
    "max_new_tokens": int(os.environ.get("GARNET_BALANCED_MAX_NEW_TOKENS", "100")),
    "ignore_eos": int(os.environ.get("GARNET_BALANCED_IGNORE_EOS", "0")),
}


def parse_json_object(text):
    candidate = text.strip()
    if candidate.startswith("```json"):
        candidate = candidate[len("```json"):].lstrip()
    elif candidate.startswith("```"):
        candidate = candidate[3:].lstrip()
    if candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
    parsed = json.loads(candidate)
    if isinstance(parsed, list):
        assert len(parsed) == 1, parsed
        parsed = parsed[0]
    assert isinstance(parsed, dict), parsed
    return parsed

cold_start = time.perf_counter()
cold = model.forward(request)
assert cold["status"] == "ok", cold
cold_ms = (time.perf_counter() - cold_start) * 1000.0
assert int(cold["visual_token_count"]) == visual_token_count, cold
assert int(cold["width"]) == 1312, cold
assert int(cold["height"]) == 736, cold
if os.environ.get("GARNET_BALANCED_PROBE_ONLY", "0") == "1":
    print(
        "Qwen VLM balanced single-request probe passed: "
        f"load_ms={load_ms:.2f}, request_ms={cold_ms:.2f}, "
        f"prompt_tokens={cold['prompt_token_count']}, "
        f"visual_tokens={cold['visual_token_count']}, "
        f"generated={cold['generated_token_count']}, "
        f"token_ids={list(cold['token_ids'])}, text={cold['text']!r}"
    )
    raise SystemExit(0)

warm_start = time.perf_counter()
warm = model.forward(request)
assert warm["status"] == "ok", warm
warm_ms = (time.perf_counter() - warm_start) * 1000.0
assert int(warm["visual_token_count"]) == visual_token_count, warm
assert str(warm["text"]).strip(), warm

frame_results = []
bbox_output_value = os.environ.get("GARNET_BALANCED_BBOX_OUTPUT_DIR", "").strip()
bbox_output_dir = Path(bbox_output_value) if bbox_output_value else None
if bbox_output_dir:
    bbox_output_dir.mkdir(parents=True, exist_ok=True)
frames = sorted((REPO_ROOT / "data" / "Dataset.1980Love" / "imgs").glob("*.jpg"))[:4]
assert len(frames) == 4
for frame in frames:
    frame_request = dict(request)
    frame_request["image_path"] = str(frame)
    frame_start = time.perf_counter()
    frame_result = model.forward(frame_request)
    assert frame_result["status"] == "ok", frame_result
    assert int(frame_result["visual_token_count"]) == visual_token_count, frame_result
    if request["ignore_eos"]:
        assert int(frame_result["generated_token_count"]) == request["max_new_tokens"], frame_result
    else:
        assert int(frame_result["generated_token_count"]) <= request["max_new_tokens"], frame_result
    raw_text = str(frame_result["text"])
    item = {
        "frame": frame.stem,
        "ms": (time.perf_counter() - frame_start) * 1000.0,
        "generated": int(frame_result["generated_token_count"]),
        "ttft_ms": float(frame_result["time_to_first_token_ms"]),
        "decode_ms": float(frame_result["decode_ms"]),
        "decode_tokens_per_second": float(frame_result["decode_tokens_per_second"]),
        "prompt_tokens": int(frame_result["prompt_token_count"]),
        "text": raw_text,
        "token_ids": list(frame_result["token_ids"]),
    }
    if bbox_output_dir:
        parsed = parse_json_object(raw_text)
        assert isinstance(parsed.get("description"), str), parsed
        persons = parsed.get("persons")
        assert isinstance(persons, list), parsed
        assert len(persons) <= 1, persons
        image = Image.open(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        boxes = []
        for person_index, person in enumerate(persons):
            assert isinstance(person, dict), person
            bbox = person.get("bbox")
            assert isinstance(bbox, list) and len(bbox) == 4, person
            assert all(isinstance(value, (int, float)) for value in bbox), bbox
            x1, y1, x2, y2 = (float(value) for value in bbox)
            assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000, bbox
            pixel_box = [
                round(x1 * image.width / 1000),
                round(y1 * image.height / 1000),
                round(x2 * image.width / 1000),
                round(y2 * image.height / 1000),
            ]
            line_width = max(3, image.width // 320)
            draw.rectangle(pixel_box, outline=(255, 48, 48), width=line_width)
            label = str(parsed["description"]).strip() or f"person {person_index + 1}"
            label_box = draw.textbbox((pixel_box[0], pixel_box[1]), label)
            label_height = label_box[3] - label_box[1] + 8
            label_top = max(0, pixel_box[1] - label_height)
            draw.rectangle(
                [pixel_box[0], label_top, pixel_box[0] + label_box[2] - label_box[0] + 8, pixel_box[1]],
                fill=(255, 48, 48),
            )
            draw.text((pixel_box[0] + 4, label_top + 2), label, fill=(255, 255, 255))
            boxes.append({"normalized": bbox, "pixels": pixel_box})
        annotated_path = bbox_output_dir / f"{frame.stem}_bbox.jpg"
        image.save(annotated_path, quality=95)
        item["json"] = parsed
        item["boxes"] = boxes
        item["annotated_image"] = str(annotated_path)
    frame_results.append(item)

if bbox_output_dir:
    (bbox_output_dir / "results.json").write_text(
        json.dumps({"prompt": request["prompt"], "results": frame_results}, indent=2),
        encoding="utf-8",
    )

print(
    "Qwen VLM balanced 943-visual-token generation passed: "
    f"state={status['state']}, partitions={status['engine_partition_count']}, "
    f"load_ms={load_ms:.2f}, engine_prepare_ms={float(status['engine_prepare_ms']):.2f}, "
    f"cold_ms={cold_ms:.2f}, warm_ms={warm_ms:.2f}, "
    f"prompt_tokens={warm['prompt_token_count']}, generated={warm['generated_token_count']}, "
    f"text={warm['text']!r}"
)
for frame_result in frame_results:
    print(
        f"{frame_result['frame']}: {frame_result['ms']:.2f} ms, "
        f"generated={frame_result['generated']}, "
        f"ttft_ms={frame_result['ttft_ms']:.2f}, "
        f"decode_ms={frame_result['decode_ms']:.2f}, "
        f"decode_tokens_per_second={frame_result['decode_tokens_per_second']:.2f}, "
        f"text={frame_result['text']!r}"
    )
