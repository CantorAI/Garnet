# Optional single-row coder vocabulary kernel

The full 2x decode / half-TTFT goal remains unmet. This native experiment is
default OFF and is not selected: actual coder serving throughput is flat and
first-token latency is slightly worse. It follows the preceding
single-row BF16 component trials: the coder head direction repeats across four
ABBA rounds, while other head/body directions are mixed. Component timing is
not inference throughput.

Latest selected-bundle telemetry records coder B1 decode GPU time 12.6723 ms
and CPU encoding 0.4893 ms per step. GPU and wait overlap and are not added.
The candidate therefore tests a GPU vocabulary projection rather than inferring
a large serving benefit from command-encoding savings.

`GARNET_METAL_SINGLE_CODER_HEAD` creates a second pipeline from the original
`bf16_decode` shader with only `p.rows==2` replaced by compile-time false.
Original first-row float operations, raw BF16 learned bytes, address space,
128 threads and 16 outputs/group stay unchanged. Only rows=1, K=896,
N=151936 select it. Batch two, text/VL and body shapes retain their original
routes. Constant-address head is a separate incompatible option. Joined MLP,
residual/packed norm and reset experiments are disabled in this trial.

Context Begin/Abort and session clear/reset/run/failure reset the new counter.
The session publishes it after successful GPU submission. Dense workers record
actual counts for the prefill prediction and every subsequent decode prediction;
the VL worker checks its final zero count for its different head shape. The
future native comparison requires count=1 only for candidate coder B1, count=0
elsewhere, complete sequence parity and unchanged fresh-image identities.

## Actual build, validation and native performance

Fourteen candidate files were installed only after thirteen prior source hashes
matched branch base `fc653485`; all originals remain preserved. The initial
build failed on a C++ test declaration that grouped `MetalBF16Weight` and
`MetalMatrix` under one `auto`. Initial PID 79789 exits 1 after 26.25 seconds.
No numerical result or benchmark is claimed from that failed attempt.

The repair splits that single declaration. Its fourteen prior source identities
match the failed candidate; no kernel, input, numerical bound or assertion
changes. A fresh directory preserves the failed build. The repaired candidate
`build-single-coder-head-repaired/libgarnet.dylib` SHA-256 is
`ec927190e2830e0e8045fd6f18e74a1d90373d21fc6165355b2fe0a16e8abde0`.

The repaired build and both CTests pass on the actual M4. The new test exercises
the complete 151,936-column output with four fresh input variants, bit parity
with the original batch-two first row, lowest-index argmax, independent FP64
gamma(ceil(K/32)+32)*sumabs+1e-7 bounds, original batch-two/body fallback,
Begin/Abort/readiness and recovery. Seventeen exact synthetic row patterns
permit reuse of independently computed FP64 references while checking every
GPU output; trained checkpoint coverage remains required in the broader suite.

Owned PID **80953** completed the full nineteen-stage original validation at
`ROOT/logs/single-coder-head-repair-checks-20261014`, exiting 0 in 1198.86 seconds.
All eight complete original dense-tail fixtures per dense model, all three
private-KV suites, serving/images, all model layers and heads, vision and
extended prefill passed. Historical partial snapshots remain preserved.

Dependent PID **86463**, directory `ROOT/logs/single-coder-head-native-20261014`,
verified that exact owned validation PID/command before waiting and started
all six sequential native ABBA profiles only after terminal validation exit 0.
It completed successfully in 1043.69 seconds including that dependency wait.
Each profile retains two fresh workers per policy and four warm non-first
requests. First-request TTFT and initialization stay separate, OS caches are
unflushed, output caps/prompts/images and resolution remain frozen, and no
inference runtimes run concurrently. Native control library remains
`887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f`.

The full local verifier checked all **678 archived files**, all nineteen
accepted original stages/cases and monitors, both CTests, source/binary
identities, the failed/repaired attempts and the exact declaration-only repair.
All **72 complete matching requests** retain the frozen prompt IDs, generated
tokens, finished flags and fresh image hashes. Candidate coder B1 has one actual
specialized head dispatch for the prefill prediction and every subsequent
decode prediction; all other candidate profiles and all controls have zero.
Warm means and distinct first-use phases are independently recomputed.

| Model / batch | Control -> candidate tokens/s per row | Control -> candidate warm TTFT ms | Decode ratio |
| --- | ---: | ---: | ---: |
| Text / 1 | 57.9355 -> 59.4725 | 48.7760 -> 50.3756 | 1.0265 |
| Text / 2 | 57.0099 -> 56.8369 | 81.6611 -> 81.6755 | 0.9970 |
| Coder / 1 | 71.9031 -> 72.0559 | 32.4879 -> 32.7852 | 1.0021 |
| Coder / 2 | 70.7408 -> 70.3378 | 50.9109 -> 50.8906 | 0.9943 |
| VL / 1 | 47.7269 -> 47.7038 | 195.2312 -> 194.7469 | 0.9995 |
| VL / 2 | 44.5319 -> 44.0734 | 361.8506 -> 360.8716 | 0.9897 |

Coder B1 decode differs by +0.21%, warm TTFT is +0.92% worse and first request
AFTER initialization is 124.69 -> 129.71 ms. The prior repeated component
signal did not produce a useful serving gain. Text B1's apparent +2.65% occurs
with zero specialized dispatches and is not a coder-kernel gain; other
nonselected profiles show mixed variation. No repeat is scheduled merely to
seek more favorable numbers. The candidate stays optional and unselected.

First request after initialization control -> candidate TTFT ms is text B1
158.51 -> 157.77, text B2 196.22 -> 189.23, coder B2 136.39 -> 141.36,
VL B1 342.55 -> 333.91 and VL B2 558.08 -> 561.46. Initialization is separately
recorded in `native/verification.json`. OS/driver caches were not flushed;
there is no observed hardware-cold sum or cold-speedup claim.

Archive SHA-256:
`1b4c785b03de1854da4aaffb1ca0ae312e57d70025fd1251dc573930adefbb57`.
Full precision data, actual per-prediction counts and phase telemetry are in
`native/verification.json` and `native/prefill-phases.json`; exact logs,
commands and source bundles are in `native/evidence`. This is a native-control
comparison, with no fresh vLLM study or 2x/half-TTFT claim. Selected libraries
and protected GPT-OSS/Pi checkouts remain unchanged.

```text
python add_single_coder_head_counters.py
python prepare_single_coder_head_sources.py
python3 install_single_coder_head_sources.py ROOT
python3 launch_single_coder_head_checks.py ROOT ROOT/logs/single-coder-head-checks-20261014 checks
python prepare_single_coder_head_repair.py
python3 install_single_coder_head_repair.py ROOT
python3 launch_single_coder_head_repair_checks.py ROOT ROOT/logs/single-coder-head-repair-checks-20261014 checks
python prepare_single_coder_head_profiles.py
python3 queue_single_coder_head_profiles.py ROOT ROOT/logs/single-coder-head-native-20261014
python3 archive_single_coder_head_native.py ROOT ARCHIVE
python verify_single_coder_head_native.py RESULTS REPORT_JSON
```

Counter preparation is a one-time editing helper and rejects reapplication to
the completed source. Reproduction should use the archived source bundles or
committed implementation and preserve the pinned control. The strict 0.5-second
guard rejects other inference/probe jobs; shorter external activity between
samples cannot be categorically excluded.
