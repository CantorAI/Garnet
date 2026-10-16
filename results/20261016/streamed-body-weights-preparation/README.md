# Streamed decoded weight lifetime preparation

Status at the preparation timestamp: Windows-only source preparation and structural preflight. No Metal
compilation, GPU numerical identity, register count, occupancy or speed is
measured. No Mac deployment, native integration or matched-target update.
This saved preparation predates the [subsequent completed GPU experiment](../streamed-body-weights/README.md).

The accepted body kernel reconstructs four float4 weight vectors into an array,
then consumes them in increasing j order. Hypothesis: consume each reconstructed
vector immediately inside its original compact/raw branch, reducing source-level
live weight storage from four vectors to one. The compiler may already optimize
the two source forms identically. This is a candidate to test, not an established
register reduction or speed improvement.

Keep the same four outputs per SIMD, 64 threads, rows 1/2, 128-K groups,
payload/metadata/raw-fallback addresses, bit reconstruction expressions, K and j
iteration order, accumulation statements, final SIMD reductions and writes.
Duplicate the original accumulation loop inside each uniform metadata branch;
only one branch executes. This differs from the earlier two/eight-output,
threadgroup-size and explicit-FMA variants. No new packing, quantization or
activation precision is proposed.

The generated shader reverses to the original text after undoing these explicit
source moves and entry renames. Generate an isolated native context/header with
an actual dispatch counter and original control, plus a draft numeric/safety
harness. The first harness-generation preflight expected three identical policy
loop strings but found two plus a differently named warm loop. Correct only the
occurrence assertion and the separate warm policy bound. No GPU or numerical
acceptance test had executed. [Source preparation record](preparation.json).

A separate prepared probe adds numeric/timing modes. Preserve original exact
output bits, FP64 gamma envelope, finite results and comparison tolerances.
Numerical mode aborts on a failed original gate; a future runner must finish all
three models' numerical runs before any timing mode. Keep original tail,
fallback, prefill, ownership and abort/recovery safety cases, with fixed 64-thread
geometry. The earlier draft harness retains legacy timing text and is not the
proposed calibrated timing protocol.

Prepared timing mode covers all six first-layer projection-set profiles, a
2-second common warmup per profile, own-null before comparison, 10 ABBA rounds,
3 warm calls and 5 raw samples/window. Expected total is 2,400 samples, all
retained. No sample exists yet; the exclusive runner, compiler validation,
independent audit and fixed admission screen must precede a real experiment.
[Separate-mode preparation](calibrated-preparation.json).

Prepared files and their source snapshots are byte-recorded in the
[manifest](manifest.json). This records source provenance only. Actual Metal
compiler scheduling and FP32 contraction can differ despite preserved source
expressions; original GPU output-bit checks remain mandatory. Any future model
comparison must retain full prompts, limits, outputs, numerical gates and the
two-repeat matched-vLLM target.

Preparation ran locally while the private width64 prefill native checks and
controllers consumed frozen Mac sources. Neither that running tree nor its
shared tools were changed by this work. The full 2x decode/half-TTFT goal remains
active.
