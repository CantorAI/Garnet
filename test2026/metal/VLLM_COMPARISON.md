# Garnet versus vLLM-Metal: 2× target

The active target is at least twice vLLM-Metal's decode steps per second and
at most half its first-token latency on the same Apple M4 Mac mini (16 GiB,
macOS 15.6.1). Results must come from actual repeated inference. The previous
Garnet-versus-Garnet improvement is not evidence of beating vLLM-Metal.

Start with Qwen3-0.6B and Qwen2.5-Coder-0.5B-Instruct, using the existing local
BF16 checkpoints. Preserve the small models as regression controls. Larger
text/coder checkpoints and Qwen3.5 image inference will be evaluated after
the matched baseline and compatibility checks.

The frozen initial workloads are:

| Model | Prompt | Batch | Output cap | State/context capacity |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | What is the capital of France? Answer briefly. | 1 and 2 | 64 | 256 |
| Qwen2.5-Coder-0.5B-Instruct | Write a Python function add(a, b) that returns their sum. Output only code. | 1 and 2 | 64 | 256 |

Garnet exports its exact native prompt token IDs for vLLM-Metal to consume.
Both runtimes use greedy generation with the same stop tokens, no prompt-prefix
reuse and no speculative decoding. The vLLM comparator uses its default
execution policy, not an artificially slower eager path. Both load the identical
checkpoint files; hashes and package versions are recorded. Activation/KV
precision and memory accounting differences must also be disclosed:
Garnet currently retains FP32 activations/KV and accumulation while keeping
decoder projections/head in BF16; vLLM-Metal executes a BF16 model/KV profile.
This is a matched checkpoint/weight-precision comparison, not a claim that all
internal arithmetic is bit-identical.

The optional `mps` prefill policy expands those BF16 checkpoint values exactly
into FP32 MPS matrices once, while decode/head keep their BF16 storage. It
changes representation and memory use, not the checkpoint values. The default
`shared` policy keeps the existing shared BF16 prefill projections. Case JSON
and native output record the selected policy; first-use conversion/compilation
and warm timing are kept separate. This experiment does not change vLLM's
execution policy or dtype.

`GARNET_METAL_PROFILE_OPS=1` enables serialized per-operation GPU diagnostics.
This splits command submissions and is excluded from all speed comparisons.

The VL profile uses Qwen3.5-0.8B, RealWorld-04, 288x192 RGB, 54 visual tokens,
79 total prompt tokens, capacity 256, chunk 16, output cap 32 and batches 1/2.
The same isolated CPython frontend opens/resizes/normalizes each source frame,
then supplies native FP32 patches or the identical resized RGB bytes to a
persistent worker. Both frontends execute the same common preparation and save
pixel/patch hashes. vLLM's processor is constrained to the already prepared
pixel dimensions; its normal GPU vision/model path remains active. Every
row/frame has a new MM UUID, with processor/prefix caches and speculation off.
Native vision encoding runs again on every frame. GPU model work never overlaps.

Both workers communicate through owned command/event files because native CLI
stdout buffering prevents reliable streaming markers. Parent `time.monotonic`
timestamps observe small first/done events before parsing bulky diagnostics.
Common preparation, file transport, reset/cache preparation and inference are
inside image-to-result timing. Worker-only timing is retained separately.
Polling is 1 ms and is part of the observed serving interface, not sub-ms kernel
precision. Fresh-process initialization is recorded separately; filesystem and
compiler caches are not flushed. The native FP32 versus comparator BF16
vision/activation/KV difference remains explicit despite identical weights.

Run each configuration in two separate native processes, alternating
Garnet/vLLM-Metal/Garnet/vLLM-Metal. Each process handles one first-use request
and two warm requests. First-use latency and initialization are recorded
separately; the OS filesystem cache is not flushed, so these are fresh-process
measurements rather than a claim of cold storage. Warm weights never imply
reused KV/vision state. Compare first
prediction time and decode calls after the first prediction using monotonic
clocks; save every engine step so stop-token accounting can be checked.
Garnet's per-request state reset is inside the request timer, as is vLLM's
request submission and cache preparation; loaded weights do not exclude that work.
Decode throughput uses the entire wall interval from first to final prediction,
including time between calls, since vLLM engine workers may execute asynchronously.
Individual call durations remain diagnostics, not the primary speed denominator.
Batch 2 duplicates the same prompt; report per-row and aggregate throughput.

Only one inference runtime may be alive at a time. A process must finish and
release its model before the next runtime starts. Do not run builds, reference
tests or package installation during measurements. Save process pre/postflight
snapshots, commands, exit statuses, outputs, checkpoint hashes and versions.
Runtime/model changes require fresh matched measurements.
The wrapper records high-water individual-process RSS; vLLM may create engine
workers, so this is not aggregate worker memory or unique unified-memory use.

Install comparator packages inside `garnet-metal-investigation/comparators/`.
Keep Garnet's SDK/runtime and the GPT-OSS/Pi checkouts untouched. An unmet
target remains unmet in the report; neither lower weight precision, a smaller
checkpoint, repeated-prompt caching, shorter output nor a slowed comparator
counts toward the target.
