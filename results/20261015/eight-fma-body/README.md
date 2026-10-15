# Explicit-FMA eight-output decode: correct, rejected after repetition

The explicit FP32 FMA transformation restored the required original output bits
in these fixtures. It did not produce a repeatable component speed improvement.
This does not establish why the two older lexical/array variants changed bits;
[their numerical failures remain rejected](../coalesced-eight-body/README.md).

The candidate changes four named accumulator expressions to explicit FP32
vector `fma`, retaining compact four-output BF16 bytes, lossless expansion,
FP32 arithmetic, lane/loop order, original control and all admission gates.
These are private generated copies; serving defaults were not changed.

Initial PID38096 exited0 in20.684103s. Confirmation PID38768 exited0
in19.939030s. Each independently audited cohort passed72 original exact-bit
and FP64/native-envelope records plus189 safety profiles, then executed1,152
timed samples in the original two ABBA rounds. Both used identical runtime
source, compiler flags, original checkpoint assets, hardware/runtime versions
and binary ae68465e09125e9e18958cb6cf052e76ce97f7fb62b04377eb5c914a9ef70635.

Across both cohorts, zero of18 policies improved both wall and GPU time in
both runs. The apparent initial coder B1/256-thread wall gain (~3.5%) reversed
to a26.2% wall regression in confirmation. Reject this candidate for promotion.
There is no full-model improvement or fresh vLLM ratio.

Initial archive SHA256:
`9efd6b5b1f16e983d1ec2283697f35810b45d3d816658b55c567c2c155daf56f`.
Confirmation archive SHA256:
`f5a5cfe2440ef2adb2a4d2d4e7c8f5fab00f00cd66949ac069d2c7b668aaa8c9`.
Each archive contains52 manifest files. [Initial audit](initial/verification.json),
[confirmation audit](confirmation/verification.json),
[joined source identities and timing directions](confirmation-comparison.json).

The first Mac audit attempt used an older base auditor without the stronger
shader-regeneration assertion. After the owned jobs were terminal and archived,
the current stronger auditor was synchronized and both audits passed. No
numerical predicate was removed or relaxed. The launcher also waited for the
complete command-buffer native retry and its audit before this GPU work.

```text
python3 launch_eight_fma_body.py ROOT ROOT/logs/eight-fma-body-20261015 component
python3 archive_eight_fma_body.py ROOT ARCHIVE [TRIAL]
python -X utf8 test2026/metal/verify_eight_fma_body.py RESULTS RESULTS/verification.json
python -X utf8 test2026/metal/compare_body_probe_confirmation.py REPORT REPORT/confirmation-comparison.json
```

ROOT is /Users/zsdx/garnet-metal-investigation. Templates above supplement
the archived executed commands; they do not assert an unexecuted reproduction.
The original CPU packing proof is reused, not independently repeated. Warm
component observations are not hardware-cold or end-to-end model timings.
The full2x decode/half-TTFT goal remains unmet.
