# Native interleaved screening index: profile-dependent tradeoffs

The lossless index layout passes all native gates and 144 complete outputs
across two six-profile trials. VL decode improves about 3%, and coder B1
decode / coder TTFT improve in both runs. Text B1 decode regresses twice;
text TTFT and VL B1 TTFT worsen twice. Initialization is longer. This is
not a general serving improvement, so the option remains OFF by default.
The 2x-decode/half-TTFT goal remains unmet, with no fresh vLLM ratio here.

Both policies enable the same 1024-row exact seed, screened original BF16
refinement, ordered prefill streaming, partial reset and serial vision
prefetch. Only index layout and observation counters differ. Every derived
index value is checked through its four-column permutation before native
publication; the padded final group is zero. The macro-selected Metal body
matches the audited component functions. Index values, original learned
BF16 values, FP32 operation order, bounds, images, prompt IDs, output limits,
stop semantics and complete normalized outputs remain unchanged.

Validation PID 33362 exited 0 in 1312.2053 s. All 27 stages pass: original
19 model/cache/vision/prefill stages, trained full-grid vision, 51 trained
head queries and six recovery stage suites. Synthetic CTest uses vocabulary
8209 to exercise the partial four-column group, outside-seed lowest ties,
bounded/nonfinite/mixed fallback, ownership and Begin/Abort. Trained heads
check every original rejected score and retained score bits. Trained session
tests observe actual layout counts through normal prediction, float-head
fallback, generic logits, nonfinite poison, reset, close/rebuild and isolation.
Every serving candidate prediction records actual layout dispatch count 1;
legacy controls report the frozen row-major configuration. Both policies
record seed width 1024, one screened head per prediction and 12 vision
prefetch dispatches per freshly encoded image request.

Initial native PID 33417 exited 0 in 1470.9927 s including the dependency
wait. Fresh repeat PID 59868 exited 0 in 160.0026 s. Independent audits check
1,861 archived files and 144 complete requests. Each profile uses fresh
sequential control/candidate/candidate/control workers, three requests per
worker. The first request is separate; four warm requests per policy form
each mean. No competing inference runtime is intentionally run; guards sample
every 0.5 s. Both image policies use the same combined threaded decode /
exact striped resize and fresh encoding without prefix/encoder reuse.

| Model / batch | Initial decode control / candidate tokens/s per row | Repeat decode control / candidate tokens/s per row |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 64.570 / 63.186 | 65.053 / 63.235 |
| Qwen3-0.6B / 2 | 61.400 / 61.696 | 61.506 / 61.379 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 78.960 / 80.042 | 78.575 / 79.938 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 76.900 / 76.689 | 76.866 / 77.133 |
| Qwen3.5-0.8B VL / 1 | 52.950 / 54.506 | 52.371 / 54.174 |
| Qwen3.5-0.8B VL / 2 | 48.482 / 49.972 | 48.355 / 50.047 |

| Model / batch | Initial TTFT control / candidate ms | Repeat TTFT control / candidate ms |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 34.986 / 37.497 | 35.664 / 36.264 |
| Qwen3-0.6B / 2 | 55.818 / 57.794 | 56.559 / 58.771 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 27.859 / 26.485 | 27.698 / 26.476 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 44.512 / 42.229 | 44.395 / 42.217 |
| Qwen3.5-0.8B VL / 1 | 173.765 / 182.603 | 170.560 / 185.338 |
| Qwen3.5-0.8B VL / 2 | 326.293 / 319.256 | 325.181 / 323.453 |

VL B1/B2 decode gains are 2.94% / 3.07% initially and 3.44% / 3.50% on
repeat. Coder B1 gains 1.37% / 1.73%. Text B1 loses 2.14% / 2.79%.
Text B2 and coder B2 decode directions change. Coder B1/B2 TTFT improve
4.93% / 5.13% initially and 4.41% / 4.91% on repeat. Text and VL B1 TTFT
regress in both runs; VL B2 TTFT improves 2.16% / 0.53%. The standalone
10.6–17.4% head-component gain does not represent a whole-model gain.

Initialization is 0.16–0.36 s longer across the paired measurements. The
candidate performs packing and an exhaustive permutation check before GPU
use, but these intervals do not isolate packing from other initialization.
First request after initialization and peak individual child RSS are stored
separately in both verification files and phase summaries. RSS is not unique
Metal memory; overlapping clocks must not be summed. Caches were not flushed,
so the first-use samples are not hardware-cold observations. No cause for all
profile-dependent variations is established by these measurements.

The next latency study holds this exact library and logical VL B2 input fixed
while retuning internal prefill chunks. It independently checks first/last
trained layers and carries state under the original numerical envelope before
timing any changed plan. That study is separate from these frozen-profile
measurements and does not create a new vLLM comparison. The subsequent
[reference-gated retuning](../replanned-vl-prefill/README.md) rejects chunk16
and gives repeated 1.9–3.0% VL B2 TTFT reductions for chunk64, with effectively
flat decode. Its 36 complete outputs and original numerical envelopes pass.

Candidate library SHA-256:
`837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471`.
Control library SHA-256:
`d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876`.
Initial archive SHA-256:
`4724c2b609d838a37d18e583287fec04a635088ca6f79faee22f527e5ada80a1`.
Combined archive SHA-256:
`718c5bab11bc3f9c4a2898b679e8df109f1e3856b34e0e403500abca8456685c`.

```text
python3 launch_interleaved_head_checks.py ROOT ROOT/logs/interleaved-head-checks-20261015 checks
python3 queue_interleaved_head_profiles.py ROOT ROOT/logs/interleaved-head-native-20261015
python3 archive_interleaved_head.py ROOT ARCHIVE
python -X utf8 verify_interleaved_head.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_interleaved_head.py ROOT ROOT/logs/interleaved-head-native-confirm-20261015
python3 archive_interleaved_head.py ROOT ARCHIVE interleaved-head-native-20261015 interleaved-head-native-confirm-20261015
python -X utf8 verify_interleaved_head.py EVIDENCE CONFIRMATION_JSON interleaved-head-native-20261015 interleaved-head-native-confirm-20261015
```
