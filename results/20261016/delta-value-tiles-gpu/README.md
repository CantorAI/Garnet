# Wider delta value tiles GPU component investigation

Status: two complete unchanged GPU cohorts and their fixed selection pass
identical independent Mac/Windows audits. Only width-64 prefill is confirmed.
No native integration or full-model speed improvement is established.

Hypothesis: share the existing Q/K norm prefix and barriers across additional
value bands per group. The original tile is already 16 columns per 128-thread
group. Widths 32/64 retain every original per-value float4 expression while
using more registers. Decode and prefill are separate selections; this does
not reuse the rejected extra-dispatch norm cache.
[Earlier CPU address prerequisite](../delta-value-tiles-addresses/README.md).

## Setup failure and correction

Initial process 92974 exits 1 in 1.358539291 seconds before compilation or GPU
execution: the reused exclusivity guard expects the alias `output`, absent
from the new runner. Freeze all 19 source/metadata/log files before defining
`output=out`. Pre-execution fixture review also catches the original maker's
inactive final token when extending learned cases to chunk 1. Make that token
active for chunk 1 and assert its actual asset values before execution;
longer learned chunks retain an inactive final token. No numerical tolerance
is involved. [Preserved setup failure](setup-failure/archive-verification.json).

The corrected run is separately named `delta-value-tiles-gpu-guardfix-20261016`.
Before launch, require the prior native norm/RoPE cohort and controller to be
terminal and archived, and finish evidence transfers. Use a separate private
tool directory. Freeze its consumed sources until terminal and archived.

## Numerical coverage

Process 93093 exits 0 in 90.803874875 seconds including compilation, fixture
preparation, numerical checks and timing. Compile with Apple Clang and Metal;
the standalone control is the original recurrence with only its entry renamed.
Match accepted d52's main-shader default compile options; native-bfloat-body
is disabled. Regenerate shader and fixture-maker transforms independently.

Generate 104 fixtures: 96 synthetic cases across batches 1/2, heads 2/16,
chunks 1/2/16/32/64/80 and ordinary/inactive/correlated/zero-decay modes, plus
8 learned cases at actual first/last linear layers 0/22, batches 1/2 and chunks
1/80. The frozen NumPy maker uses actual trained QKV, convolution, decay and
beta weights with synthetic hidden/prior states. These are component inputs,
not full model requests or learned projections executed by Garnet.

For every case, check padding 0/4 floats, three carried state steps and widths
16/32/64. Change Q each step. Compare output and next-state words bit for bit
against the original kernel; check read-only inputs, inactive outputs, finite
results and state/output padding guards. All 208 profiles/1,872 checks pass.
The 553,986,648 FP32 word comparisons include padding guards and width-16
control self-checks. Native graph/session ownership and full trained-model
validation remain required.
[Raw numeric/clock audit](initial/verification.json),
[identical Mac audit](initial/mac-verification.json).

## Timing and fixed screen

Use all eight learned endpoints, both candidate widths, 16 repeated primitive
dispatches per command, a 2-second common warmup per width, then own-null
before comparison. Each has 10 ABBA rounds, 3 warm commands and 5 raw timing
samples per window, 6,400 samples total. Keep every raw GPU and wall interval.
Repeated fixture inputs here calibrate the primitive only; full-model request
comparisons prohibit prefix/input/image reuse.

Compute candidate/control ratios of arithmetic means of all 100 raw samples
per policy at each endpoint. Do not normalize by nulls or filter samples.
The runner records endpoint/threshold rules before execution. Mean aggregation
was implemented after launch and before timing values were inspected; this
is not a formal preregistered study.
Require both own-null GPU/wall fractions within [.95,1.05] and both comparison
fractions <=.95 at every one of four endpoints per width/workload. Endpoints
are first/last layer crossed with B1/B2, separately for decode chunk 1 and
prefill chunk 80. Preserve all initial failures through confirmation.

All 16 initial own-nulls are stable. Width-64 prefill passes all four endpoints;
the other three width/workload selections fail. Its component GPU fractions
range .869049–.915350 and wall fractions .876546–.912910. Decode B2 is slower
for both widths; width-32 prefill misses the threshold at layer 22/B1.
These are initial kernel measurements, not TTFT reductions.
[Full initial selection](initial/analysis.json),
[identical Mac selection](initial/mac-analysis.json).

Confirmation 93509 started only after initial archive/audit and all original
source/checkpoint/config identities match. Repeat every numeric case, width,
endpoint and timing sample; do not rerun just favorable endpoints. The initial
fixed selection is only width-64 prefill. No favorable third repeat is allowed.

## Unchanged confirmation result

Confirmation 93509 exits 0 in 90.376146083 seconds. All source, generated
shader/fixture driver, checkpoint/config, assets and measurement-design
identities match. Both independent hosts reproduce every numeric/clock record
and the same two-cohort selection. Across both cohorts there are 416 numeric
profiles, 3,744 output/state checks and 1,107,973,296 compared FP32 words,
including control self-checks and guards, plus 12,800 raw timing samples.
[Repeat audit](repeat/verification.json),
[repeat endpoint analysis](repeat/analysis.json),
[fixed two-cohort decision](decision-repeat.json).

Only width-64 prefill passes every first/last x B1/B2 endpoint in both cohorts.
Repeat GPU fractions range .873431–.906314, wall .880689–.903724. The 32-column
candidate and both decode selections remain unselected. No third repeat,
source promotion or fresh vLLM benchmark is performed.

All endpoint fractions below are candidate/control. Null entries give initial /
repeat stability for both clocks; passing requires >=5% gain on both clocks.

| Width | Layer | Batch | Chunk | GPU initial / repeat | Wall initial / repeat | Stable null initial / repeat | Pass initial / repeat |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 32 | 0 | 1 | 1 | 0.837997 / 0.872979 | 0.924096 / 0.967068 | Yes / Yes | Yes / No |
| 64 | 0 | 1 | 1 | 0.903053 / 0.894000 | 0.997451 / 0.947142 | Yes / Yes | No / Yes |
| 32 | 0 | 1 | 80 | 0.947907 / 0.946383 | 0.949062 / 0.942059 | Yes / Yes | Yes / Yes |
| 64 | 0 | 1 | 80 | 0.869049 / 0.877305 | 0.876546 / 0.885098 | Yes / Yes | Yes / Yes |
| 32 | 0 | 2 | 1 | 1.096702 / 1.143955 | 1.062057 / 1.097111 | Yes / Yes | No / No |
| 64 | 0 | 2 | 1 | 1.281286 / 1.307722 | 1.193874 / 1.221114 | Yes / Yes | No / No |
| 32 | 0 | 2 | 80 | 0.885651 / 0.855905 | 0.889206 / 0.858254 | Yes / Yes | Yes / Yes |
| 64 | 0 | 2 | 80 | 0.910351 / 0.906314 | 0.908295 / 0.903724 | Yes / Yes | Yes / Yes |
| 32 | 22 | 1 | 1 | 0.874008 / 0.833194 | 0.935409 / 0.925540 | Yes / Yes | Yes / Yes |
| 64 | 22 | 1 | 1 | 0.957123 / 0.887098 | 1.005156 / 0.957624 | Yes / Yes | No / No |
| 32 | 22 | 1 | 80 | 0.967522 / 0.951755 | 0.968278 / 0.956876 | Yes / Yes | No / No |
| 64 | 22 | 1 | 80 | 0.876493 / 0.873431 | 0.881957 / 0.880689 | Yes / Yes | Yes / Yes |
| 32 | 22 | 2 | 1 | 1.098192 / 1.113783 | 1.076526 / 1.072218 | Yes / Yes | No / No |
| 64 | 22 | 2 | 1 | 1.269558 / 1.265538 | 1.189247 / 1.168949 | Yes / Yes | No / No |
| 32 | 22 | 2 | 80 | 0.841809 / 0.863726 | 0.844865 / 0.864926 | Yes / Yes | Yes / Yes |
| 64 | 22 | 2 | 80 | 0.915350 / 0.891772 | 0.912910 / 0.889651 | Yes / Yes | Yes / Yes |

The initial width-64 prefill control costs .325971–.330683 ms per dispatch at
B1 and .625915–.629425 ms at B2, computed by dividing each 16-dispatch GPU
command interval by 16. Control-minus-candidate differences are only
.040260–.043303 ms at B1 and .053281–.056112 ms at B2 in this isolated workload.
This is not per-frame latency or measured whole-model savings. Kernel gains
must not be substituted for the last matched comparison's roughly 28% VL TTFT gap. Integrate
behind a guarded native policy only after native correctness/ownership tests,
then measure complete model TTFT/decode/setup/memory and fresh comparators.

## Reproduction and archive identities

ROOT=`/Users/zsdx/garnet-metal-investigation`, private tools at
`ROOT/delta-value-tiles-component-tools-20261016`:

```text
python3 TOOLS/launch_delta_value_tiles_gpu.py ROOT ROOT/logs/delta-value-tiles-gpu-20261016
python3 TOOLS/launch_delta_value_tiles_gpu.py ROOT ROOT/logs/delta-value-tiles-gpu-guardfix-20261016
python3 TOOLS/launch_delta_value_tiles_gpu_confirmation.py ROOT ROOT/logs/delta-value-tiles-gpu-confirm-20261016
python3 ARCHIVE_TOOLS/archive_delta_value_tiles_gpu.py ROOT ROOT/logs/delta-value-tiles-gpu-guardfix-20261016 ROOT/delta-value-tiles-gpu-guardfix-20261016.tar.gz
python -X utf8 test2026/metal/verify_delta_value_tiles_gpu.py results/20261016/delta-value-tiles-gpu/initial results/20261016/delta-value-tiles-gpu/initial/verification.json
python -X utf8 test2026/metal/analyze_delta_value_tiles_gpu.py results/20261016/delta-value-tiles-gpu/initial results/20261016/delta-value-tiles-gpu/initial/analysis.json
```

| Evidence | SHA256 |
| --- | --- |
| Setup failure before GPU execution, 19 files | `bcc1ebb205c13aa0f6dd1313eb199df0e2f3fcbedf74dc2ca285abeca624816b` |
| Initial complete component cohort, 36 files | `59c92a01e6353b28270a06b48d8d801a632d6d5628075bb83e0a9c4bb999c867` |
| Unchanged complete confirmation, 38 files | `a60f360cd8e4ba0609218c189ea25b4ea457b28ea581d9a6a6e5d7c694b72812` |

The archive retains raw source/logs/config/asset manifest and hashes of 312
binary input files plus the executable. Wider tiles add no logical state or
weight tensors; their register occupancy and isolated pipeline properties are
recorded in raw logs, not a measured full-model memory saving. Native source,
models and matched-vLLM counts remain unchanged.

```text
python -X utf8 test2026/metal/compare_delta_value_tiles_gpu_cohorts.py results/20261016/delta-value-tiles-gpu results/20261016/delta-value-tiles-gpu/decision-repeat.json
```
