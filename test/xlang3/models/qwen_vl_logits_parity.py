# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose the WorldFusion Qwen case using identical teacher-forced tokens.

Run native with the CPython 3.14 bridge, reference with the isolated Torch Python.
All model, cache and output paths are explicit. No reference inference fallback.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent / "WorldFusion"))
from pipeline.semantics import DEFAULT_PROMPT, INPUT_SHAPES, load_garnet


def native(args, reference):
    import xlang3
    model, handles, status = load_garnet(ROOT, args.weights, args.cache)
    g = xlang3.importModule("garnet", fromPath=str(ROOT.parent / "out/build/x64-Release/bin/garnet.dll"))
    b = xlang3.importModule("builtins")

    def convert(value):
        if isinstance(value, list):
            result = b.list()
            for item in value:
                result.append(convert(item))
            return result
        if isinstance(value, dict):
            result = b.dict()
            for key, item in value.items():
                result[key] = convert(item)
            return result
        return value

    def tensor(values, dtype="int64", shape=None):
        array = np.asarray(values)
        return g.tensor_from_host(convert(array.reshape(-1).tolist()), dtype=dtype,
                                  shape=convert(list(shape or array.shape)))

    def cpu(value):
        return np.asarray(list(g.tensor_to_cpu(value).tolist()))

    p = g.qwen_vl_prepare_request(model_dir=str(args.weights), image_path=str(args.image),
                                 prompt=DEFAULT_PROMPT, min_pixels=256 * 28 * 28,
                                 max_pixels=1280 * 28 * 28)
    pixels = np.load(args.pixels, allow_pickle=False)
    np.testing.assert_array_equal(cpu(p["pixel_values"]).reshape(pixels.shape), pixels)
    ids = list(p["input_ids"])
    assert ids == reference["input_ids"], "Prompt token mismatch"
    length = len(ids)
    profile = INPUT_SHAPES[0][1]
    padded = np.zeros((1, profile), dtype=np.int64)
    padded[0, :length] = ids
    types = np.zeros_like(padded)
    types[0, :length] = list(p["mm_token_type_ids"])
    mask = np.zeros_like(padded)
    mask[0, :length] = 1
    positions = np.zeros((3, 1, profile), dtype=np.int64)
    positions[:, :, :length] = cpu(p["position_ids"]).reshape(3, 1, length)
    delta = int(cpu(p["mrope_position_deltas"]).reshape(-1)[0])
    tensors = xlang3.importModule("tensor")
    key = g.tensor_to_gpu(tensors.tensor(shape=convert(INPUT_SHAPES[11]), dtype=tensors.bfloat16))
    value = g.tensor_to_gpu(tensors.tensor(shape=convert(INPUT_SHAPES[12]), dtype=tensors.bfloat16))
    pages = tensor(np.arange(INPUT_SHAPES[13][0]), "int32")
    inputs = [tensor(padded), g.tensor_to_bfloat16(p["pixel_values"]),
              tensor(reference["image_grid_thw"]), p["vision_bilinear_indices"],
              p["vision_bilinear_weights"], p["vision_position_ids"], p["vision_cu_seqlens"],
              tensor(types), tensor(mask), tensor(positions), p["mrope_position_deltas"],
              key, value, pages, tensor([0], "int32")]
    result = model.forward({"inputs": convert(inputs), "sample": "greedy", "sample_row": length - 1})
    assert result["status"] == "ok", str(result)
    assert int(result["token_id"]) == reference["token_ids"][0]
    decoder = g.load_model(str(ROOT / "xModel/qwen3/vl_2b_instruct/qwen_text_decode.py"),
                          runtime_mode="compiled_xmodel", entry_function="Qwen3TextDecode",
                          weights=str(args.weights),
                          input_shapes=convert([[1, 1], [3, 1, 1], INPUT_SHAPES[11],
                                                INPUT_SHAPES[12], INPUT_SHAPES[13], [1], [1]]),
                          input_dtypes=convert(["int64", "int64", "bfloat16", "bfloat16",
                                                "int32", "int32", "int32"]),
                          cache_dir=str(args.cache / "decode"))
    assert decoder.runtime_status()["ready"], str(decoder.runtime_status())
    for step in range(1, args.step + 1):
        slot = length + step - 1
        inputs = [tensor([[reference["token_ids"][step - 1]]]),
                  tensor(np.full((3, 1, 1), slot + delta, dtype=np.int64)),
                  key, value, pages, tensor([slot + 1], "int32"), tensor([slot], "int32")]
        result = decoder.forward(convert({"inputs": inputs, "sample": "greedy",
                                          "return_logits": int(step == args.step)}))
        assert result["status"] == "ok", str(result)
    logits = cpu(result["output"]).astype(np.float32).reshape(-1)
    np.save(args.output, logits)
    print("native", "step", args.step, "token", int(result["token_id"]), flush=True)


def reference_run(args, reference):
    import torch
    from transformers import LogitsProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.weights, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    if args.fp32_head:
        # The head is tied to embeddings; do not change embedding precision.
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.float(), requires_grad=False)
        model.lm_head.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(),))
    pixels = torch.from_numpy(np.load(args.pixels, allow_pickle=False)).to("cuda")
    inputs = dict(input_ids=torch.tensor([reference["input_ids"]], device="cuda"),
                  attention_mask=torch.ones((1, len(reference["input_ids"])), dtype=torch.long, device="cuda"),
                  pixel_values=pixels,
                  image_grid_thw=torch.tensor(reference["image_grid_thw"], device="cuda"))
    class SharedPrefix(LogitsProcessor):
        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - len(reference["input_ids"])
            if step < args.step:
                forced = torch.full_like(scores, -float("inf"))
                forced[:, reference["token_ids"][step]] = 0
                return forced
            return scores

    with torch.inference_mode():
        if args.generate_tokens:
            result = model.generate(**inputs, do_sample=False, max_new_tokens=args.generate_tokens)
            tokens = result[0, len(reference["input_ids"]):].tolist()
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.weights, local_files_only=True)
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            args.output.write_text(json.dumps({"token_ids": tokens, "fp32_head": args.fp32_head,
                                                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                                                "eos_reached": tokens[-1] in eos}, indent=2))
            print("reference generation", len(tokens), "last token", tokens[-1], flush=True)
            return
        result = model.generate(**inputs, do_sample=False, max_new_tokens=args.step + 1,
                                logits_processor=[SharedPrefix()], output_logits=True,
                                return_dict_in_generate=True)
    logits = result.logits[-1][0].float().cpu().numpy()
    np.save(args.output, logits)
    print("reference", "step", args.step, "token", int(logits.argmax()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["native", "reference"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=31)
    parser.add_argument("--fp32-head", action="store_true")
    parser.add_argument("--generate-tokens", type=int, default=0)
    parser.add_argument("--compare-native", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    assert 0 < args.step < len(reference["token_ids"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (native if args.mode == "native" else reference_run)(args, reference)
    if args.compare_native:
        assert not args.generate_tokens and args.mode == "reference"
        actual = np.load(args.compare_native, allow_pickle=False)
        expected = np.load(args.output, allow_pickle=False)
        assert actual.shape == expected.shape and actual.ndim == 1
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        top = np.argsort(actual)[-8:][::-1]
        report = {"step": args.step, "reference_fp32_head": args.fp32_head,
                  "native_token": int(actual.argmax()), "reference_token": int(expected.argmax()),
                  "rmse": float(np.sqrt(np.mean((actual - expected) ** 2))),
                  "max_abs_error": float(np.max(np.abs(actual - expected))),
                  "cosine": float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected))),
                  "top_native": [{"token": int(i), "native": float(actual[i]),
                                  "reference": float(expected[i])} for i in top]}
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
