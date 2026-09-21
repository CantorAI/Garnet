# Qwen-VL Test And Parity Plan

## Test Pyramid

```text
HF reference smoke
  -> processor dump tests
  -> Garnet import/load tests
  -> weight/config validation
  -> subgraph numeric parity
  -> end-to-end logits parity
  -> decode text parity
  -> serving latency tests
```

## Phase 04 Tests

Current files:

- `test2026/tests/phase_04_qwen_vl_reference/test_hf_qwen_vl.py`
- `test2026/tests/phase_04_qwen_vl_reference/test_garnet_qwen_vl.py`

Default behavior:

- skip with exit code `0`
- no accidental model downloads
- no hard dependency on `numpy`, `torch`, `transformers`, or `xlang` unless the test is enabled

## Prompt Suite

The current prompt is a smoke prompt:

```text
Describe this movie frame briefly. The detector metadata contains N boxes;
mention visible people, objects, and scene context.
```

Next prompt suite should be explicit:

| Prompt ID | Prompt | Output Type |
| --- | --- | --- |
| `caption_short` | Describe the scene in one sentence. | text |
| `people_count` | How many people are visible? Answer with a number and short explanation. | structured text |
| `object_check` | Are there vehicles, bicycles, or traffic signs visible? | yes/no plus evidence |
| `event_verify` | Does this frame show a person entering, leaving, waiting, or interacting with an object? | classification plus explanation |

Reference outputs should record:

```json
{
  "prompt_id": "caption_short",
  "prompt": "...",
  "answer": "...",
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "image": "...",
  "device": "cuda",
  "dtype": "bf16",
  "latency_ms": 123.4
}
```

## Processor Dump

The reference test should support:

```powershell
$env:HF_QWEN_VL_DUMP_PROCESSOR="1"
```

Dump artifacts:

- tokenized input IDs
- attention mask
- image tensor/pixel values metadata
- image grid metadata
- chat template string
- prompt ID

Large tensor dumps should be opt-in.

Suggested output folder:

```text
test2026/artifacts/qwen_vl_reference/
```

Artifacts may be ignored by git unless explicitly needed.

## Garnet Parity Flow

Early flow:

```text
read HF processor dump
load Garnet DLL through xlang
load qwen_vl_model.py
load config/weights if available
create Garnet tensors from processor dump
run first supported stage
compare or report exact unsupported stage
```

The test should never fail with vague messages like:

```text
NoneType object is not callable
```

Instead it should fail/skip with:

```text
unsupported stage: qwen3_vl_merge_visual_embeddings
missing weight: visual.blocks.0.attn.qkv.weight
missing DLL: D:/.../garnet.dll
```

## Numeric Tolerances

Initial suggested tolerances:

| Mode | rtol | atol |
| --- | --- | --- |
| fp32 debug | `1e-4` | `1e-5` |
| fp16 | `5e-2` | `5e-2` |
| bf16 | `8e-2` | `8e-2` |

For logits, compare:

- top-k overlap
- max absolute error
- cosine similarity
- optional exact greedy next token

Do not require full text match as the first parity criterion.

## Latency Test Shape

Once correctness exists, add:

```text
single image / single prompt
single image / multiple prompts
fresh image sequence
text-only decode
mixed fresh-image and decode workload
```

Metrics:

- preprocess ms
- vision ms
- merge ms
- prefill ms
- TTFT
- per-token decode ms
- GPU memory
- output tokens/sec

## CI Policy

Default CI should run:

- syntax/compile checks
- skip-safe Phase 04 tests
- small unit tests

Explicit GPU/model CI should run only when:

- model weights are present
- GPU is present
- env flags are set

This protects local development while allowing serious benchmark runs.
