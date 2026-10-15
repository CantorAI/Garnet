# Body-only fixed-shape serving experiment

Status: fresh build, standard Metal MPS CTest, all 19 original numerical stages,
six trained session/recovery cases and all six native ABBA profiles passed.
Independent verification covers 1594 archived files and 144 complete matching
requests, exact body dispatch counts, zero fallback misses, zero specialization
in the frozen prefill profiles, and bounded eager pipeline caches. Initial
decode gains are small and broad, and persist in a second six-profile trial.
Batch-2 TTFT is higher in both trials, so the default remains OFF.
The full 2x/half-TTFT goal remains unmet. No fresh vLLM ratio is claimed.

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

Both jobs terminated with exit 0. Validation took 1206.3553 seconds and the
queued recovery/native job took 1334.2174 seconds including its dependency wait.
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

## Actual initial native comparison

Four warm requests per policy/profile, from two fresh sequential workers.
Each worker's first request is excluded from warm means. Decode is tokens/s per
row; batch-2 aggregate rate is twice that value. These compare native builds.

| Model | Batch | Control decode | Body-shape decode | Decode ratio | Control TTFT ms | Body-shape TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 59.2460 | 60.7312 | 1.02507 | 48.4984 | 48.4635 |
| Qwen3-0.6B | 2 | 57.7762 | 59.3273 | 1.02685 | 81.9444 | 83.7880 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.0166 | 72.8867 | 1.01208 | 33.5652 | 32.9447 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 71.1275 | 71.3922 | 1.00372 | 51.5568 | 51.7298 |
| Qwen3.5-0.8B VL | 1 | 48.1461 | 48.8642 | 1.01492 | 198.9916 | 199.2997 |
| Qwen3.5-0.8B VL | 2 | 44.4502 | 45.0202 | 1.01282 | 361.3436 | 362.0627 |

All six profiles completed on attempt 0. Text B2 warm TTFT is 2.25% higher in
this run despite no specialized prefill dispatch; this observation must not
be hidden by the decode improvement. Initialization and first request after
initialization are reported separately in `verification.json` and `phases.json`.
The phase report preserves pipeline counts, per-decode dispatch totals and
fallback counts. No pipeline compilation is deferred into a measured decode.

Initial archive SHA-256:
`3f737b51044f1c18f003c6e4a11545378b5041a3c3c212f2d636b7672362aaf5`.
The independent initial audit was uploaded before launching confirmation
PID 13690 at `logs/body-shape-confirm-native-20261015`. The repeat pins the
exact candidate dylib and installed source manifest and retains the same guards.

## Confirmation and decision

Confirmation PID 13690 exited 0 in 144.5687 seconds. It completed all six profiles
on attempt 0. Both independent audits passed all original stages, trained
recovery cases, complete output sequences and exact route counters. Per profile,
the prepared pipeline counts are 5 for text, 4 for coder and 8 for VL. Every
decode call specializes exactly 196, 168 and 186 body projections respectively,
with zero misses, at both batch sizes. Frozen prefill counts remain zero.

| Model | Batch | Control decode | Body-shape decode | Decode ratio | Control TTFT ms | Body-shape TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 59.3267 | 60.9131 | 1.02674 | 50.9812 | 48.6984 |
| Qwen3-0.6B | 2 | 57.8497 | 59.2837 | 1.02479 | 79.4604 | 82.7190 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 71.9080 | 73.1748 | 1.01762 | 32.7509 | 32.0319 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.5210 | 70.5973 | 1.00108 | 50.7192 | 51.7019 |
| Qwen3.5-0.8B VL | 1 | 47.8962 | 48.5413 | 1.01347 | 195.0254 | 195.1461 |
| Qwen3.5-0.8B VL | 2 | 44.3396 | 45.1892 | 1.01916 | 359.7874 | 363.9336 |

Decode improves in both trials for all profiles, from 0.1% to 2.7%; coder B2
is almost flat. Text B2 warm TTFT rises 2.25%/4.10%, coder B2 0.34%/1.94%,
and VL B2 0.20%/1.15%. No specialized prefill dispatch occurs in these workloads;
the reason for these latency changes has not been established. This is a
measured optional decode optimization with a latency tradeoff, not a completed
2x/half-TTFT solution. Keep the default OFF and the selected global-alpha
serving library unchanged while targeting the remaining latency gap.

Initialization and first-use behavior remains mixed across profiles. Text B1's
first request after initialization is slower in both trials (155.7272→158.2436
and 155.1636→157.9324 ms). Eager compilation is included in initialization and
does not appear as a fallback compilation during measured decode. Full values
are retained separately in both verification and phase reports; caches were
not flushed, so these are not hardware-cold measurements.

Combined archive SHA-256:
`5c5ff519f09633d00a42987dee826e81c55c84fc99db31c8c7be119ab83e4c0c`.
`verification.json` and `confirmation-verification.json` independently recompute
their respective 72 complete requests from this combined manifest. Both phase
reports retain initialization, first-use, warm clocks and actual cache counts.
The original GPT-OSS and Pi checkouts remain clean. No fresh vLLM comparison
was run for this optional candidate, and older comparator measurements must not
be combined with these native numbers as a new runtime ratio.
