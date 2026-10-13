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

Run each configuration in two separate native processes, alternating
Garnet/vLLM-Metal/Garnet/vLLM-Metal. Each process handles one first-use request
and two warm requests. First-use latency and initialization are recorded
separately; the OS filesystem cache is not flushed, so these are fresh-process
measurements rather than a claim of cold storage. Warm weights never imply
reused KV/vision state. Compare first
prediction time and decode calls after the first prediction using monotonic
clocks; save every engine step so stop-token accounting can be checked.
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
