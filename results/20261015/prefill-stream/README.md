# Ordered partial prefill submissions: repeated latency tradeoff

This optional experiment starts GPU prefill work while the CPU encodes later
layers. Existing telemetry records about 5–6 ms host prefill encoding before
the original single command buffer is submitted. The experiment aims to
overlap that work. Two actual native ABBA runs reduce warm first-token latency
in all six profiles, but do not establish a decode speed gain. Keep this
experiment optional and default OFF. There is no fresh vLLM comparison.

`GARNET_METAL_PREFILL_STREAM` defaults OFF. Only resident prefill with chunk
greater than one submits after each four layers, excluding the last boundary.
Single-token decode keeps one submission. Model kernels, learned values,
precision, prompts, image bytes/resolution, caps, stopping rules and frozen
chunk plans remain unchanged. The candidate retains global half alpha; dirty
prefix reset, vision alpha and other experimental optimizations remain off.

MetalContext retains every submitted command and all pending output owners.
Pending values are usable by their producing context but unavailable for host
readback/update. Final submission waits for all command buffers and checks
every status before publishing readiness. Abort drains submitted work, drops
the uncommitted continuation and invalidates all produced outputs before
reuse. Session graph locks remain held through completion. Actual partial
submission counts are published only after a successful full operation.

The design uses one queue and tracked shared resources. Apple's documentation
describes [command ordering](https://developer.apple.com/documentation/metal/mtlcommandbuffer)
and [automatic resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization).
No kernel is reordered within the logical dependent sequence. GPU telemetry
reports the earliest GPU start to latest GPU end across the command group,
including inter-command gaps; it does not sum potentially overlapping spans.
GPU time can overlap both CPU encoding and CPU waiting. Phase intervals must
not be added to infer end-to-end latency or isolated kernel work.

Installation pinned nine exact original sources against `cfc051a6`, preserved
their bytes and installed ten candidate files. Isolated candidate library
SHA-256 is
`bad8c5db120df0e6ac107a0fabc6b19708a379f36785b7b5c8869a30fd14534b`.
Configure/build and both CTests pass on the actual M4. The new unit checks 27
chains across rows 1/2/3, columns 1/17/1024 and 1/3/7 partial submissions,
with whole-output bit equality against one command. It verifies pending host
access rejection, ownership retention, abort draining and buffer reuse, count
reset and a real MPS-to-compute dependency across multiple submissions. The
original numerical CTest remains unchanged.

Validation PID **98669** exited 0 after 1201.14 seconds: all nineteen original
model/cache/head/image/prefill gates and both CTests passed. Dependent PID
**3161** exited 1 before any inference request because its helper looked up
the validation metadata under date 20261014 instead of 20261015. Preserve
that failure in `failed/` and `failure-verification.json` (351 verified files,
zero native requests). The repair changed only the comparator metadata path;
the candidate binary, kernels and numerical gates were unchanged.

Repaired PID **20856** exited 0 after 145.01 seconds; repeat PID **23797**
exited 0 after 144.83 seconds. Each ran all six sequential native ABBA
profiles with 72 complete requests and four warm requests per policy/profile
from two fresh workers, excluding each first request. Independent audits
verify 941 combined archive files, all nineteen gates and all 144 complete
matching serving outputs. Fresh frame hashes, prompt IDs and finished flags
match. Actual candidate partial counts are 6 for text, 5 for coder and VL B1,
and 15 for VL B2; control and decode counts are zero. Guards reject competing
inference/probe jobs and only stop owned work.

| Model / batch | Initial TTFT control → candidate ms | Repeat TTFT control → candidate ms | Decode candidate/control initial / repeat |
| --- | ---: | ---: | ---: |
| Text / 1 | 49.508 → 45.158 | 50.380 → 44.506 | 1.0107 / 0.9851 |
| Text / 2 | 79.009 → 76.574 | 82.105 → 77.731 | 1.0019 / 0.9913 |
| Coder / 1 | 33.118 → 29.339 | 32.709 → 28.659 | 0.9822 / 0.9950 |
| Coder / 2 | 51.857 → 46.119 | 50.984 → 46.657 | 0.9855 / 0.9946 |
| VL / 1 | 201.617 → 190.993 | 197.364 → 190.522 | 0.9997 / 0.9977 |
| VL / 2 | 359.370 → 348.868 | 360.874 → 350.313 | 1.0032 / 0.9985 |

Repeated warm TTFT reductions are 11.7%, 5.3%, 12.4%, 8.5%, 3.5% and 2.9%.
Decode is 0.1–1.5% slower in the repeat; text's initial positive direction did
not repeat. The method is a latency tradeoff, not a throughput optimization.
Repeat first request AFTER initialization improves in every profile (text
158.518→129.841 / 188.769→165.571 ms; coder 127.850→106.672 /
139.929→116.341 ms; VL 335.761→316.931 / 563.676→528.435 ms).
Initialization is recorded separately and OS/driver caches are not flushed.

`phases.json` and `confirmation-phases.json` retain the actual clocks. Repeat
text B1 prefill falls 42.366→37.038 ms while GPU span is 34.775→34.988 ms;
host wait falls 35.764→30.563 ms. This supports overlapping host encoding,
not a claim of faster matrix kernels. Reset remains unchanged by this method.
Do not add independent dirty-prefix-reset gains to these measurements; a
combined build requires its own numerical, recovery and serving evidence.

Combined archive SHA-256:
`23994803ed9d97238b0b870a82880e98520fc299dbba6116166d7ec1816850a0`.
Initial-only archive SHA-256:
`2bf7fe3e9d5994b456d4fcb429f448a3b8c46deebf0429373f9fe03934393e89`.
Failed archive SHA-256:
`320b263e883fa44f2e54436f853e85cb4fe3519d8247cbfb1870c41e60c3c0a4`.

`pending/` is a historical bounded snapshot of actual handles, source and
library hashes, successful build/CTest stages and real unit stdout. Remaining
full-model validation and serving timings are now terminal and archived in
`evidence/`, `verification.json` and `confirmation-verification.json`.
The original selected libraries and GPT-OSS/Pi checkouts remain untouched.
The full three-model 2x decode / half-TTFT goal remains active and unmet.

```text
python prepare_prefill_stream_sources.py
python3 install_prefill_stream_sources.py ROOT
python3 launch_prefill_stream_checks.py ROOT ROOT/logs/prefill-stream-checks-20261015 checks
python3 queue_prefill_stream_profiles.py ROOT ROOT/logs/prefill-stream-native-20261015
python3 launch_repaired_prefill_stream_profiles.py ROOT OUTPUT
python3 launch_confirm_prefill_stream_profiles.py ROOT OUTPUT
python3 archive_prefill_stream_repaired_native.py ROOT ARCHIVE
python verify_prefill_stream_repaired_native.py EVIDENCE VERIFICATION_JSON
python verify_prefill_stream_repaired_native.py EVIDENCE CONFIRMATION_JSON --confirm
python summarize_prefill_stream_phases.py JOBS PHASES_JSON
```
