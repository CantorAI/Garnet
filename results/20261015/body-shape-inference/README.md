# Body-only fixed-shape serving experiment

Status: fresh build and standard Metal MPS CTest passed. All 19 original model
validation stages are running; six trained session/recovery cases and six native
ABBA profiles are queued behind the exact owned validation job. No whole-model
performance result is available. The full 2x/half-TTFT goal remains unmet.

The preceding two component trials found repeated small body gains for coder B2
and VL, while text/coder vocabulary-head specialization consistently slowed
those heads. This candidate tests whether body gains survive real graphs while
the agreed models' original heads remain unchanged. It does not enable the
reset, streaming, fused-head, concurrent-pass, or native-bfloat experiments.

`GARNET_METAL_BODY_SHAPE` defaults OFF. It retains original BF16 learned bytes
and FP32 arithmetic/inputs/outputs. Graph initialization prepares pipelines for
physical rows 1 or 2 and output width below 100000. All three frozen Qwen
vocabulary heads exceed this width and retain their original route. Larger
prefill matrices use the existing MPS path. Tiny prefill profiles with at most
two physical rows are also eligible by shape; the six frozen benchmark prefill
plans all have more than two physical rows.

Each specialized pipeline fixes dimensions and physical strides. Tail branches
are removed only for complete output tiles. A shared, mutex-protected cache is
bounded to 128 profile entries and releases with the last owning context.
Compilation occurs before a graph becomes ready and contributes to model init.
Decode performs a lookup and does not compile lazily. Missing/unsupported
entries retain original dispatch; benchmark admission requires zero misses.

Workers record the count after prefill and every decode call. Dense layers
must use seven specialized body projections and VL linear layers eight. The
frozen prefill count must be zero, all decode counts must equal the exact layer
configuration total, and the bounded cache count must be positive. Every
complete output, prompt ID, stop flag and fresh image encoding is checked.

Trained one-layer session cases cover eager cache preparation, original greedy
heads and generic full logits, private state, reset, nonfinite poisoning and
recovery, close/rebuild, and stable cache size across repeated runs. All original
19 numeric validation stages remain required before timing begins.

Actual owned jobs:

| Job | PID | Directory |
| --- | ---: | --- |
| Fresh build and all original gates | 88191 | logs/body-shape-checks-20261015 |
| Dependency wait, recovery, native profiles | 89188 | logs/body-shape-native-20261015 |

Both handles were confirmed live after launch. Inspect process state and
exit.json before any continuation; a launch record is not completion evidence.
Candidate dylib SHA-256 after the successful build:
`e39bbdc6f731412b003f66aa1efe6f83bc71faecd4eef205dad399a728531d21`.
The selected control library and GPT-OSS/Pi checkouts are unchanged.

```text
python3 install_body_shape_sources.py ROOT
python3 launch_body_shape_checks.py ROOT ROOT/logs/body-shape-checks-20261015 checks
python3 queue_body_shape_profiles.py ROOT ROOT/logs/body-shape-native-20261015
```

After terminal success, archive with `archive_body_shape_inference.py` and audit
with `verify_body_shape_inference.py`. A useful initial result still requires a
fresh six-profile repeat. Initialization and first request after init remain
separate from warm averages; compiler and OS caches are not flushed. New native
numbers must not be paired with the older vLLM numbers as a fresh comparison.

## Existing VL phase evidence guiding further work

The older selected-control confirmation records show the following warm means.
These are baseline observations, not results of this candidate. The source is
`../reused-zero-source-inference/evidence/logs/reused-zero-source-confirm-native-20261015/jobs`.

| Batch | Total TTFT ms | Common preprocessing ms | Worker TTFT ms | Reset + vision ms | Assembly ms | Prefill wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 193.4979 | 45.8409 | 145.1534 | 58.6511 | 0.3091 | 86.1933 |
| 2 | 359.9487 | 48.2508 | 308.8741 | 124.2580 | 0.5210 | 184.0951 |

The worker's field named vision_seconds starts before reset, so it includes
reset and must not be added to reset_seconds again. Worker TTFT includes its
reset/vision, assembly and prefill subintervals. Total TTFT also contains common
preprocessing and protocol overhead; its nested clocks must not all be summed.
These observations indicate substantial remaining image/vision/prefill work.
They do not establish an optimization or allow changing pixels, output caps,
precision or other matched-workload requirements.
