# Three exact half-storage activation planes: prefill feasibility

Status: CPU representation study executed; synthetic MPS lower-bound probe
prepared and queued. No GPU precision, model compatibility or speed claim yet.

Motivation: retained-control prefill GPU spans are83.781/144.813ms for VL
B1/B2; text/coder also spend substantial TTFT in prefill. Existing native
common-alpha products use FP32 activations and exact half-stored BF16 weights.
The rejected BF16-feed study rounded source activations. This study instead
asks whether three half-storage pieces can reconstruct source FP32 bits, and
whether the cost of three products leaves useful headroom before decomposition.
This is a feasibility investigation; no algorithmic novelty is claimed.

Represent a bounded normal input by `h0=half(x)`,
`h1=half((x-float(h0))*2^11)`, and
`h2=half((x-float(h0)-float(h1)*2^-11)*2^22)`.
Require finite planes and exact original source-bit recovery; otherwise a future
implementation must retain the original FP32 route. This runtime admission and
fallback are not implemented by the CPU study or the synthetic timing probe.

## Actual CPU representation result

[CPU raw source/plane hashes, cases and versions](cpu-reconstruction.json).
The Windows CPU script exits0 and verifies2,826,240 source inputs over three
model gate input widths, physical rows80/160 and four exact power-of-two scales.
All these bounded synthetic source values reconstruct bit for bit. They are
generated from the same bounded normal bit pattern as the proposed MPS probe,
not captured model activations.

Independently sample255 finite exponent fields, nine selected mantissas and
both signs:4,590 edge inputs. Only769 reconstruct exactly;3,821 require the
original FP32 fallback, including unsupported dynamic ranges/signed-zero cases.
These samples do not estimate real model fallback frequency or universal FP32
coverage. Nothing is clipped or silently rounded into admission.

Final prospective MPS probe source SHA256:
`847f1dc136dcbab13ff537b600ca575e3a7b174fe789bfe692dbc2a2fd01baee`.
The CPU calculation was repeated after adding diagnostic actual-value and
warmup/buffer reporting to that unqueued probe; numeric reconstruction and
all admission counts remain unchanged. No CPU timing benefit is claimed.

```text
python -X utf8 test2026/metal/analyze_half_triplet_reconstruction.py results/20261016/half-triplet-lower-bound/cpu-reconstruction.json
```

## Prospective synthetic MPS lower-bound probe

Use the six actual MLP gate shapes from the frozen model configurations:
inner/output1024/3072 text,896/4864 coder,1024/3584 VL, each with80/160 rows.
Weights are structured nonzero synthetic dyadic values, not learned weights.
Inputs and all three half planes are precomputed on the host, outside timing.
No universal source conversion, learned checkpoint or native inference claim.

Control: one FP32-input/half-weight product, FP32 output. Candidate: three
half-input/half-weight products with alpha1/2^-11/2^-22 and beta0/1/1 into the
same FP32 output. Same-control-pipeline null is a third mode. Dense products
execute through MPSMatrixMultiplication. No claim about MPS internal accumulator
instructions follows from output dtype. Three finite diagnostics distinguish
rounded-half products from observed FP32 products, plus one exact alpha/beta
test requiring all three source pieces to contribute. Any exception or
diagnostic failure stops timing admission and is archived as a negative result.
API reference: [Apple MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication).
The API page/spec fetch did not establish an internal accumulator guarantee;
none is assumed. Earlier explicit Metal mixed-accumulator diagnostics are a
different API and do not prove this MPS route.

For all six shapes/four input scales, require source-bit reconstruction,
complete finite outputs and padding preservation, unchanged independent FP64
gamma(ceil(K/32)+32)*sumabs+1e-7 AND5e-6+5e-5*abs(reference) envelopes,
and original control-comparison bounds. Numerics precede timing. Conditioning
is2s alternating common warmup; ten complete ABBA rounds, five samples/window,
three window warm calls, both candidate and identical-control null; all records
retained. Capture weight/setup/check duration and logical buffer byte counts
separately. They are not RSS or an inference initialization measurement.

This is an optimistic cost feasibility probe: GPU decomposition, general
exponent handling, native ownership/fallback, trained weights/activations and
full outputs remain unimplemented/unverified. Even a large synthetic gain
cannot establish the full FP32 inference goal or a vLLM comparison. No native
selection or confirmation rule is inferred after inspecting these timings.

Queue PID83151 watches exact compact component controller73358 and requires its
terminal/archive/audit success before quiet guarded dispatch. It archives one
terminal probe success/failure and audits successful raw data; it does not
search for favorable repeats. Existing vector-native jobs stay untouched and
serial. All six consumed helpers are frozen while queued.

```text
python3 ROOT/source/test2026/metal/queue_mps_half_triplet_lower_bound.py ROOT ROOT/logs/mps-half-triplet-lower-bound-queue-20261016 73358
```

ROOT is `/Users/zsdx/garnet-metal-investigation`; exact compile/run/archive/audit
commands are captured when executed. No compilation or GPU result exists yet.

Original queue83151 exits1 before launching this probe because the upstream
native audit-path failure stopped compact queue73358. All are preserved in the
vector-native45-file252aeda0... failure archive. Dependency-only repaired
wrappers resume as new queue94813 at
`logs/mps-half-triplet-lower-bound-queue-repaired-20261016`, watching compact
queue94685. Probe source, diagnostics, source reconstruction and sampling are
unchanged; the CPU evidence still pins847f1dc1.... No failed capability test is
being retried: no capability process has yet launched. All old statuses remain.
