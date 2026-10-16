# Exact common-scale union: all learned feed-forward pairs

CPU eligibility prerequisite only; no inference speed or GPU arithmetic claim.
The original codec and an independent NumPy implementation inspect all 76
gate/up pairs (152 matrices) from the three frozen checkpoints. The common
scale comes from the maximum magnitude across both matrices; no learned value
is rounded, clipped or omitted. The original C++ codec restores the admitted
values to their original FP32 bits.

| Model | Pairs | Prior equal-scale eligible | Union eligible | Required original fallback |
| --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 28 | 17 | 28 | None |
| Qwen2.5-Coder-0.5B-Instruct | 24 | 13 | 23 | Layer 8 |
| Qwen3.5-0.8B | 24 | 0 | 24 | None |

Original-codec checks restore 561,512,447 individual values exactly and reject
one value. Its entire coder layer-8 pair must fall back. Fully admitted pairs
contain 552,796,160 values. NumPy takes 28.6250 seconds; independent C++ checking,
including compilation, takes 3.7440 seconds. These are host checks, not inference.

Pairing alone leaves the logical half payload unchanged. A safe prototype that
retains the original independent buffers adds an estimated 336 MiB for text,
382.375 MiB for coder and 336 MiB for VL if all admitted pairs are prepared.
Those are payload estimates, not measured process RSS or memory savings.

The terminal 23-file archive is independently verified on Mac and Windows:
SHA256 `f753aebd6121680da5ca162bb1caed5661f304068ba6f23608e6cc5c5755e466`.
[Full evidence](evidence/manifest.json), [audit](verification.json).
Commands, offsets, checkpoint and source hashes are retained in the evidence.
The overall matched-vLLM goal remains unmet.
