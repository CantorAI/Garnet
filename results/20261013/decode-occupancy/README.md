# Decode work distribution: component evidence, not a model speed claim

Two actual isolated runs on the M4 Mac mini test four alternatives to the
selected BF16 kernel's four outputs per SIMD group / 128 threads per group:
two outputs / 64 or 128 threads, and four outputs / 64 or 256 threads.
The variants retain the original per-output FP32 accumulation and SIMD
reduction order, original BF16 bits and row-sharing for batch 2.

Each run validates and times 272 cases: actual first-layer projections and
complete runtime vocabulary heads for all three models, plus irregular
synthetic tails, at batches 1 and 2. All 544 checks are bit-identical to the
selected kernel, including output padding. Timings use repeated ABBA order,
three warm-ups and twelve samples per policy position. Only one GPU probe or
inference runtime runs at once. These are warm matrix-component measurements;
they do not represent model decode throughput or vLLM performance.

The two-output / 64-thread variant has a repeatable small-matrix signal:

| Actual matrix | Batch | Initial GPU speedup | Repeat GPU speedup |
| --- | ---: | ---: | ---: |
| Coder key projection, 128×896 | 1 | 1.636× | 1.609× |
| Coder key projection, 128×896 | 2 | 1.583× | 1.589× |
| Coder value projection, 128×896 | 1 | 1.652× | 1.563× |
| Coder value projection, 128×896 | 2 | 1.618× | 1.599× |
| Qwen3.5 `in_proj_a`, 16×1024 | 1 | 1.697× | 1.512× |
| Qwen3.5 `in_proj_a`, 16×1024 | 2 | 1.524× | 1.405× |
| Qwen3.5 `in_proj_b`, 16×1024 | 1 | 1.409× | 1.370× |
| Qwen3.5 `in_proj_b`, 16×1024 | 2 | 1.519× | 1.522× |

These save approximately 2–5 microseconds per small projection in the repeat.
Large matrices show mixed results and the complete heads do not demonstrate
a consistent gain. Therefore the variants have not been integrated into
serving, and cannot support a 2× whole-model claim. A bounded small-output
dispatch remains a possible later experiment with full-model validation.

`probe-source/` retains the actual driver, generator, asset collector and
runner. Both job directories preserve commands, terminal exits, generated
Metal source, matrix hashes, process guards and raw samples. Raw matrix
copies are excluded from the archive. The initial metadata accidentally
records the older input-tile runner's digest; that script and the initial
runner are both retained explicitly. The repeat corrects self-hashing and
retains its actual native source. Both runs use the same decode arithmetic.
