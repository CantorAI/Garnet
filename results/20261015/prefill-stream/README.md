# Ordered partial prefill submissions: validation in progress

This optional experiment starts GPU prefill work while the CPU encodes later
layers. Existing telemetry records about 5–6 ms host prefill encoding before
the original single command buffer is submitted. The experiment aims to
overlap that work; no latency, throughput or vLLM gain is yet established.

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

Owned validation PID **98669** is running all nineteen original model/
cache/head/image/prefill gates in `ROOT/logs/prefill-stream-checks-20261015`.
Dependent PID **3161**, `ROOT/logs/prefill-stream-native-20261015`, verifies
that exact live handle before waiting for terminal validation exit 0. It then
runs all six sequential native ABBA profiles with 72 complete requests. Every
candidate request must record `floor((layers-1)/4)` partial submissions per
chunk greater than one, while control and decode counts must remain zero.
All complete output/prompt IDs, flags and freshly encoded frame hashes must
match. Guards reject competing inference/probe jobs and only stop owned work.

`pending/` is a historical bounded snapshot of actual handles, source and
library hashes, successful build/CTest stages and real unit stdout. Remaining
full-model validation and serving timings require terminal evidence. Preserve
current source and candidate binaries until their archives are collected.
The original selected libraries and GPT-OSS/Pi checkouts remain untouched.
The full three-model 2x decode / half-TTFT goal remains active and unmet.

```text
python prepare_prefill_stream_sources.py
python3 install_prefill_stream_sources.py ROOT
python3 launch_prefill_stream_checks.py ROOT ROOT/logs/prefill-stream-checks-20261015 checks
python3 queue_prefill_stream_profiles.py ROOT ROOT/logs/prefill-stream-native-20261015
python3 archive_prefill_stream_native.py ROOT ARCHIVE
python verify_prefill_stream_native.py EVIDENCE VERIFICATION_JSON
```
