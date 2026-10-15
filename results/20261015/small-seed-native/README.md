# Native exact seed 1024 versus 8192: small mixed gains

The optional smaller seed passes the complete native gates and 144 complete
outputs across two trials. Some decode gains repeat, especially text B2, but
TTFT is mixed and text B1 / VL B2 regress in both runs. This does not establish
a general serving improvement or a new vLLM target ratio. The option remains
OFF by default, and the full 2x-decode/half-TTFT goal remains unmet.

Both policies use the same screened head, ordered prefill streaming, partial
cache reset and original-order vision prefetch. The candidate changes only
the exact seed width from 8192 to 1024 plus observation counters. Learned
checkpoint values, bounds, FP32 refinement, prompts, images, token caps,
stop rules, complete outputs and fresh image encoding remain frozen. Both
image policies use the same threaded decode / exact striped resize frontend.

Validation PID 99758 exited 0 in 1305.9734 s. All 27 stages pass: the original
19 model/cache/vision/prefill stages, bit-identical trained full-grid vision,
51 actual trained head queries and six trained recovery stage suites. The
head gates compare every rejected original score with the smaller actual
seed's threshold. Context tests check the actual 1024 dispatch width and
Begin/Abort recovery. Trained session tests check the published width through
normal prediction, float-head fallback, generic logits, nonfinite poison,
reset, close/rebuild and private-session isolation. All candidate predictions
in the serving trials record actual seed width 1024 and one head dispatch;
legacy controls report their frozen 8192 configuration.

Initial native PID 294 exited 0 in 1455.1244 s including the validation wait.
Fresh confirmation PID 26793 exited 0 in 160.5934 s. Independent audits check
1,845 archived files, all gates, all 144 complete outputs and every actual
seed/prefill/reset/vision counter. Each profile uses fresh sequential
control/candidate/candidate/control workers. Each worker makes three requests;
the first is separate, and the remaining four warm requests per policy form
the mean. Inference runtimes never intentionally overlap; guards sample at
0.5-second intervals. The original Mac checkout, GPT-OSS and Pi checkouts are
preserved.

| Model / batch | Initial decode control / candidate tokens/s per row | Repeat decode control / candidate tokens/s per row |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 64.160 / 64.861 | 64.120 / 64.355 |
| Qwen3-0.6B / 2 | 60.633 / 62.081 | 60.172 / 61.326 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 78.462 / 78.085 | 77.640 / 79.073 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 76.295 / 76.990 | 76.357 / 76.997 |
| Qwen3.5-0.8B VL / 1 | 53.119 / 52.810 | 52.627 / 52.773 |
| Qwen3.5-0.8B VL / 2 | 48.388 / 48.394 | 48.222 / 48.479 |

| Model / batch | Initial TTFT control / candidate ms | Repeat TTFT control / candidate ms |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 35.601 / 36.097 | 34.862 / 35.683 |
| Qwen3-0.6B / 2 | 56.203 / 57.380 | 57.350 / 56.524 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 27.763 / 28.237 | 27.960 / 27.814 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 44.920 / 44.467 | 45.170 / 44.269 |
| Qwen3.5-0.8B VL / 1 | 173.338 / 170.992 | 171.529 / 169.358 |
| Qwen3.5-0.8B VL / 2 | 323.433 / 324.999 | 323.991 / 327.757 |

Text B2 decode improves 2.39% / 1.92%, and coder B2 improves 0.91% / 0.84%.
Text B1 improves only 1.09% / 0.37% while TTFT worsens 1.39% / 2.35%.
Coder B1 and VL B1 decode directions change. Coder B2 and VL B1 TTFT improve
in both trials; text B2 and coder B1 TTFT directions change. VL B2 TTFT
worsens 0.48% / 1.16%. These are paired observations, not causal explanations
for the remaining variations. The earlier 2.4–3.7% component head reduction
must not be represented as a whole-model gain.

Initialization, first request after init, separate GPU/CPU clocks, peak
individual child RSS and actual counts are preserved in `native-phases.json`
and `confirmation-native-phases.json`. Clocks overlap and must not be summed;
RSS is not unique Metal memory. Caches were not flushed, so first use after
initialization is not a hardware-cold observation. No fresh vLLM comparison
was executed in this native-only study.

The next head study preserves index values and changes their layout. Its
[two component trials](../interleaved-screen-index/README.md) preserve masked
score/flag bits and reduce head wall time by 10.6–17.4%; full native integration
is still pending. A separate [MPS-filter route](../mps-screen-index/README.md)
passes the numerical gates but is rejected for a 53–63% wall penalty.

Candidate library SHA-256:
`d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876`.
Control library SHA-256:
`b61c4732d667ebb0c813d8d67e0b132ace611940700d211b65fb6871bcb79e0d`.
Initial archive SHA-256:
`ec064b673d715f24864992367850c0b7671f972e5dc5727493aeb3698dd9843b`.
Combined archive SHA-256:
`0ba9947d1efda8cc355fd5433bc6397c7f4c81b9b71980af85727bc6d386f6f4`.

```text
python3 launch_small_seed_checks.py ROOT ROOT/logs/small-seed-checks-20261015 checks
python3 queue_small_seed_profiles.py ROOT ROOT/logs/small-seed-native-20261015
python3 archive_small_seed.py ROOT ARCHIVE
python -X utf8 verify_small_seed.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_small_seed.py ROOT ROOT/logs/small-seed-native-confirm-20261015
python3 archive_small_seed.py ROOT ARCHIVE small-seed-native-20261015 small-seed-native-confirm-20261015
python -X utf8 verify_small_seed.py EVIDENCE CONFIRMATION_JSON small-seed-native-20261015 small-seed-native-confirm-20261015
```
