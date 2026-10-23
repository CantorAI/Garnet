# Qwen3.5 CUDA checkpoint for WorldSense

This development checkpoint adds Qwen3.5-2B and 4B CUDA execution, shared BF16
projection weights, batched profiles, allocation accounting, and opt-in packed
prefill/speculative decoding. The optimization goal continues after this commit.

The exact measured profiles, checkpoint hashes, native binary hashes, environment,
and metrics are in [the checkpoint record](research/qwen35-cuda-checkpoint.json).
Use these explicit profiles to reproduce the current path; the experimental
options are not enabled by default. The two latest real-image cohorts are:

| Model | Batch | Garnet ms | LM Studio ms | vLLM ms | GPU allocations kept by Garnet |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 1 | 186.14 | 302.21 | 203.27 | 4.749 GB |
| Qwen3.5-4B | 8 | 678.38 | 1,348.27 | 665.65 | 12.706 GB |

These are medians on 17 and 15 common equal-32-output groups respectively, from
strictly sequential RTX4080 GPU0 runs. Each Garnet cohort preserves all 28 saved
output sequences. External baselines are historical matched-input cohorts, with
different intermediate arithmetic and speculative settings. Same-build repeats
were still running when this checkpoint was recorded. The 2× target is unmet.

Across all 20 measured groups, aggregate full-request sampled throughput is
171.50 tok/s for 2B B1 and 374.47 tok/s for 4B B8. These groups differ from the
equal-32 latency subset. Tracked allocation totals count shared buffers once;
they are not process peak or physical-residency measurements.

## Using the current checkout

Build the CUDA backend with the existing XLang3 SDK and TensorRT dependencies.
The measured binary used CUDA13.2, SM89, MSVC14.51, and the default
`GARNET_QWEN35_WHOLE_PROGRAM_CUDA=OFF`. Set the environment entries from the
checkpoint record **before loading Garnet**. Keep the source checkout available:
the serving adapter imports `runtime/` and `xModel/`; standalone installation of
that adapter has not yet been validated.

Inside the XLang3 Python runtime with Garnet available, the same batched adapter
used by the benchmark can be called directly:

```python
import json
from runtime.cuda_serving import Model

with open(checkout + '/docs/research/qwen35-cuda-checkpoint.json') as stream:
    record = json.load(stream)
entry = record['profiles']['2b-b1']  # Or '4b-b8' with eight input rows.
model = Model(checkpoint_directory, entry['profile'], entry['model_id'])
try:
    result = model.infer_batch(
        ['Describe the main objects in this image in one sentence.'],
        [jpeg_path],
        32,
    )
finally:
    model.close()
```

Add the checkout to the runtime import roots. Use one prompt and image path per
row of the selected static batch. Stop and recreate the model to change its
batch, context capacity, or image grid. The measured profile uses a fixed 12×20
patch grid, 512-token slots, and a 32-token request limit. Every request resets
state and decodes/encodes its images again; prefix, image, and answer caching are
off. Larger images, longer prompts, and WorldSense camera workloads require their
own validation and measurements.

For the native single-request API, pass the model ID, checkout xModel directory,
and explicit profile to `garnet.serve_model(...)`, then use `garnet.infer_json(...)`.
The batched path above is the measured entry point. Ordinary CPython alone does
not supply the native `garnet` and `tensor` packages.

## Validation and remaining work

The current packed/native-prefix/head-reuse implementation passes 12 numerical
batch/prefill profiles, including state, stop, rollback and fault handling.
Unfiltered verifier and full 24-case batch-8 session memchecks report zero errors.
Race/sync evidence is component or prefix scoped, not full-integration coverage.
Real 2B B1 and 4B B8 probes preserve complete finite target-state hashes and
outputs against the preceding implementation and a separately ordinary target.
The native logical-view correction passes eight boundary/lifetime cases with
unfiltered memcheck, racecheck and synccheck. The current DLL also passes the
Qwen3-VL-2B two-request serving/reset/vision smoke.

Independent pretrained teacher agreement still has recorded discrepancies.
Fresh current-source Metal build preservation, optimized real-image coverage for
every batch, comparator expansion, isolated memory coverage, and standalone
packaging remain open. This checkpoint does not claim production-quality
acceptance or completion of the performance goal. Rejected D8 and NoRDC trials
remain explicit experiments.

Benchmark, reference, sanitizer-oracle, memory and analysis sources are under
`test2026/cuda/`. The detailed [report](research/qwen35-cuda-comparison-report.md)
and [development log](research/qwen35-cuda-development-log.md) retain outcomes,
failures and method limits. Raw local evidence, controller scripts, and verified
source/runtime archives remain in `D:/CantorAI/work/qwen35-cuda`; checkpoints,
images, vendor binaries, and generated engines are not included in this commit.
