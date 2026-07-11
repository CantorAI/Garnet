import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_native_prompt_image"
PHASE05_SCRIPT = REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "test_real_qwen_mlp_subgraphs.py"
DEFAULT_IMAGE = REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg"
DEFAULT_IMAGES = [
    DEFAULT_IMAGE,
    REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_1050.jpg",
    REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_10050.jpg",
    REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_10200.jpg",
]
DEFAULT_PROMPT = (
    "Detect exactly 3 distinct main visible objects when possible. Do not repeat a label. "
    "Return ONLY compact valid JSON using this schema: "
    '{"objects":[{"label":"person","bbox":[x1,y1,x2,y2],"confidence":0.95}]}. '
    "bbox coordinates must be integers normalized from 0 to 1000, with x1 < x2 and y1 < y2. "
    "confidence must be between 0 and 1. No Markdown and no text outside the JSON."
)
EXPECTED_SCENE_TERMS = {
    "frame_0.jpg": ["man", "basket", "hut", "workshop", "woven"],
    "frame_1050.jpg": ["cook", "wok", "food", "vegetable", "pepper"],
    "frame_10050.jpg": ["woman", "girl", "guitar", "room"],
    "frame_10200.jpg": ["woman", "girl", "man", "people", "room"],
}
COCO_CLASS_NAMES = {
    0: "person",
    41: "cup",
    45: "bowl",
    56: "chair",
    58: "plant",
    73: "book",
}
LABEL_ALIASES = {
    "man": "person",
    "woman": "person",
    "girl": "person",
    "boy": "person",
    "people": "person",
    "wok": "bowl",
    "pan": "bowl",
    "pot": "bowl",
    "plate": "bowl",
    "dish": "bowl",
    "books": "book",
}


def parse_detection_json(text):
    start = text.find("{")
    if start < 0:
        return None, "missing JSON object"
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list) or not objects:
        return None, "objects must be a non-empty list"
    if len(objects) > 8:
        return None, "objects exceeds the requested maximum of 8"
    normalized = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            return None, f"objects[{index}] has no string label"
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
            return None, f"objects[{index}].bbox is not four numbers"
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return None, f"objects[{index}].bbox is outside normalized range or unordered"
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            return None, f"objects[{index}].confidence is outside [0,1]"
        label = item["label"].strip().lower()
        normalized.append({
            "label": LABEL_ALIASES.get(label, label),
            "bbox": [x1, y1, x2, y2],
            "confidence": float(confidence),
        })
    return normalized, ""


def box_iou(lhs, rhs):
    ix1 = max(lhs[0], rhs[0])
    iy1 = max(lhs[1], rhs[1])
    ix2 = min(lhs[2], rhs[2])
    iy2 = min(lhs[3], rhs[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    lhs_area = max(0.0, lhs[2] - lhs[0]) * max(0.0, lhs[3] - lhs[1])
    rhs_area = max(0.0, rhs[2] - rhs[0]) * max(0.0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - intersection
    return intersection / union if union > 0 else 0.0


def reference_boxes(image_path):
    from PIL import Image

    annotation_path = image_path.with_suffix(".json")
    if not annotation_path.exists():
        return []
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        width, height = image.size
    references = []
    for bbox, class_id in zip(annotation.get("boxes", []), annotation.get("classes", [])):
        label = COCO_CLASS_NAMES.get(int(class_id))
        if label is None:
            continue
        references.append({
            "label": label,
            "bbox": [
                float(bbox[0]) * 1000.0 / width,
                float(bbox[1]) * 1000.0 / height,
                float(bbox[2]) * 1000.0 / width,
                float(bbox[3]) * 1000.0 / height,
            ],
        })
    return references


def extract_decode_json(text):
    marker = '"request_count"'
    marker_index = text.find(marker)
    if marker_index < 0:
        marker = '"processor_npz"'
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    image_list_value = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_IMAGES", "").strip()
    if image_list_value:
        images = [Path(value) for value in image_list_value.split(";") if value.strip()]
    elif "GARNET_NATIVE_PROMPT_IMAGE_TEST_IMAGE" in os.environ:
        images = [Path(os.environ["GARNET_NATIVE_PROMPT_IMAGE_TEST_IMAGE"])]
    else:
        images = DEFAULT_IMAGES
    prompt = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_PROMPT", DEFAULT_PROMPT)
    max_pixels = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_PIXELS", "65536")
    max_new_tokens = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_TEST_TOKENS", "384")
    for image in images:
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
            "GARNET_QWEN_NATIVE_ROPE_DECODE_ONE_CALL_PREPARE": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_IMAGE": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_IMAGE_PREPROCESS": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_JPEG_PREPROCESS": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_TOKENIZER": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_USE_KV_MANAGER": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_USE_CACHED_TEXT": "1",
            "GARNET_QWEN_GPU_TENSOR_CHAIN": "1",
            "GARNET_QWEN_BFLOAT16_WEIGHTS": "1",
            "GARNET_QWEN_CPP_VISION_RUNNER": "1",
            "GARNET_QWEN_CPP_TEXT_RUNNER": "1",
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_PIXELS": max_pixels,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_TOKENS": max_new_tokens,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_PROMPT": prompt,
            "GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGE": str(images[0]),
        }
    )
    if len(images) > 1:
        env["GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGES"] = ";".join(str(image) for image in images)
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
    requests = result.get("requests", [result])
    if any(not request.get("generated_text", "") for request in requests):
        raise AssertionError("Garnet native prompt-image decode produced empty generated_text")
    semantic_checks = []
    structured_checks = []
    for request in requests:
        image_name = Path(request.get("image", "")).name
        image_path = Path(request.get("image", ""))
        expected_terms = EXPECTED_SCENE_TERMS.get(image_name, [])
        generated_lower = request.get("generated_text", "").lower()
        detections, structure_error = parse_detection_json(request.get("generated_text", ""))
        detected_labels = [item["label"] for item in detections] if detections else []
        matched_terms = [
            term for term in expected_terms
            if term in generated_lower or LABEL_ALIASES.get(term, term) in detected_labels
        ]
        references = reference_boxes(image_path)
        coordinate_matches = []
        if detections:
            for detection in detections:
                candidates = [ref for ref in references if ref["label"] == detection["label"]]
                best_iou = max((box_iou(detection["bbox"], ref["bbox"]) for ref in candidates), default=0.0)
                if best_iou >= 0.1:
                    coordinate_matches.append({"label": detection["label"], "iou": best_iou})
        structured_checks.append({
            "image": image_name,
            "passed": detections is not None,
            "error": structure_error,
            "object_count": len(detections) if detections else 0,
            "detected_labels": detected_labels,
            "coordinate_match_count": len(coordinate_matches),
            "coordinate_matches": coordinate_matches,
        })
        if expected_terms:
            semantic_checks.append({
                "image": image_name,
                "expected_any": expected_terms,
                "matched": matched_terms,
                "passed": bool(matched_terms),
            })
    result["semantic_checks"] = semantic_checks
    result["semantic_pass_count"] = sum(1 for check in semantic_checks if check["passed"])
    result["semantic_check_count"] = len(semantic_checks)
    result["structured_checks"] = structured_checks
    result["structured_pass_count"] = sum(1 for check in structured_checks if check["passed"])
    result["coordinate_pass_count"] = sum(1 for check in structured_checks if check["coordinate_match_count"] > 0)
    (ARTIFACT_DIR / "last_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Phase 08: native prompt+image Garnet generated output for {len(requests)} image(s).")
    print(f"prompt={prompt}")
    for request in requests:
        print(f"image={request.get('image')} generated_text={request.get('generated_text')}")
    if "aggregate" in result:
        aggregate = result["aggregate"]
        print(
            "steady_pipeline_ms="
            f"avg:{aggregate['steady_pipeline_avg_ms']:.2f} "
            f"min:{aggregate['steady_pipeline_min_ms']:.2f} "
            f"max:{aggregate['steady_pipeline_max_ms']:.2f}"
        )
        print(
            "full_request_ms="
            f"avg:{aggregate['full_request_avg_ms']:.2f} "
            f"min:{aggregate['full_request_min_ms']:.2f} "
            f"max:{aggregate['full_request_max_ms']:.2f}"
        )
        print(f"decode_token_avg_ms={aggregate['decode_token_avg_ms']:.2f}")
    if semantic_checks:
        print(f"semantic_checks={result['semantic_pass_count']}/{result['semantic_check_count']}")
        for check in semantic_checks:
            print(f"semantic image={check['image']} passed={check['passed']} matched={check['matched']}")
    if structured_checks:
        print(f"structured_checks={result['structured_pass_count']}/{len(structured_checks)}")
        print(f"coordinate_checks={result['coordinate_pass_count']}/{len(structured_checks)}")
        for check in structured_checks:
            print(
                f"structured image={check['image']} passed={check['passed']} "
                f"objects={check['object_count']} coordinate_matches={check['coordinate_match_count']} "
                f"error={check['error']}"
            )
    print(f"result={ARTIFACT_DIR / 'last_result.json'}")
    require_semantic = os.environ.get("GARNET_NATIVE_PROMPT_IMAGE_REQUIRE_SEMANTIC", "1").strip().lower() in {"1", "true", "yes", "on"}
    failed_checks = [check for check in semantic_checks if not check["passed"]]
    failed_structured = [check for check in structured_checks if not check["passed"]]
    failed_coordinates = [check for check in structured_checks if check["coordinate_match_count"] <= 0]
    if require_semantic and (failed_checks or failed_structured or failed_coordinates):
        failed_images = sorted({
            check["image"] for check in failed_checks + failed_structured + failed_coordinates
        })
        raise AssertionError(
            "Garnet multi-image structured/semantic/coordinate validation failed for: "
            + ", ".join(failed_images)
        )


if __name__ == "__main__":
    main()
