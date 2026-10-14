# Compact joined half-weight storage: lower memory, no repeated latency gain

The existing joined-prefill experiment reduced warm TTFT by about 2–3%
in its admitted text B2 and coder B1 workloads, but retained duplicate half
weights and increased worker peak RSS by about 335/167 MiB. This candidate
lets individual projections share offset MPSMatrix views of the joined
half-weight allocation. The original learned BF16 decode buffers remain.
Per-row scale buffers are also shared, with explicit offsets. Repacking
reads both matrix and scale offsets, including a reversed weight pair.

`GARNET_METAL_COMPACT_JOINED_PREFILL` defaults OFF and requires the existing
default-OFF joined-prefill option. Admission shapes, floating-point
arithmetic, prompts, caps, batching and cache behavior are unchanged.
The selected primary and attention libraries are not replaced. This is an
allocation experiment, with no claimed decode algorithm improvement.

## Actual validation

The isolated Mac build and CTest completed with exit 0. The native joined
tests compare complete dyadic outputs bit for bit for both admission
shapes. Added tests exercise individual projections after joining,
reversed-pair repacking and individual projections after reversing. The
existing independent vision, convolution, delta, direct-write and matrix
tests also pass. All 18 guarded validation stages completed with exit 0,
including 48 trained text/coder exact-tail/state comparisons with unchanged
numerical bounds and dispatch assertions, 24 private-cache profiles,
trained VL/dense layer and output-head checks, and pretrained vision.
Pending snapshots remain historical observations; `evidence` contains the
completed job results.

Candidate library SHA-256:
`b58260e7e3a8488bf888d6f3d27a957ea209b1e28f5ee1afd12cbd4dfda1762b`.

The durable validation job PID **75480** completed with exit 0 after
847.82 seconds. Its authoritative directory is
`/Users/zsdx/garnet-metal-investigation/logs/compact-joined-checks-20261013`.
Initial benchmark PID **92146** and confirmation PID **94979** also
completed with exit 0, after 144.44 and 40.60 seconds. No inference job
from this experiment remains running.

## Actual matched native measurements

The original joined library and compact candidate ran sequentially in
ABBA order across all six frozen profiles, followed by a repeat of the
two admitted profiles. All 96 complete generated IDs, prompt IDs, finish
flags, input cases and actual joined dispatch counts match. Every VL
request freshly encodes the image. The comparison control is
the original joined library, SHA-256
`cc41fde1f3905fcfb02fed6a50ab0f9d68933e94358352b92db4e44cb5f43dfe`;
the validation build metadata separately fingerprints the unchanged
primary library. No fresh vLLM baseline has been run for this candidate.

| Admitted profile | Peak RSS reduction, initial / repeat | Warm TTFT, initial control → compact | Repeat control → compact | Repeat decode ratio |
| --- | ---: | ---: | ---: | ---: |
| Text B2 | 338.84 / 337.86 MiB | 84.99 → 83.14 ms | 85.63 → 86.07 ms | 1.00006 |
| Coder B1 | 166.86 / 165.41 MiB | 36.88 → 36.72 ms | 36.56 → 37.31 ms | 0.99452 |

The memory reduction repeats, but the small initial warm-TTFT benefit
does not. First use after initialization is slower in both runs: repeat
text B2 257.97 → 259.82 ms and coder B1 193.07 → 200.24 ms. Initialization
is separately measured and approximately unchanged: repeat text B2
1.73404/1.73462 seconds; coder B1 1.38242/1.38369 seconds. No combined
startup sum is presented as an observed clock. OS/driver caches were not
flushed; these are fresh-worker first-use measurements, not proven
hardware-cold samples. Unsupported profiles do not share joined storage;
their changes are not attributed to this method.

RSS is macOS `/usr/bin/time -l` worker/process-family peak resident set,
including initialization and first use. It is not unique GPU allocation,
an aggregate of parallel workers or per-request current memory. Two fresh
workers per policy supply four non-first warm requests per workload.
The comparison preserves arithmetic and learned values; its original
compact BF16/exact-half weight traffic differs from the investigation's
losslessly expanded FP32 vLLM comparator. This experiment establishes no
stock-BF16 vLLM advantage.

Commands, from the isolated source's `test2026/metal` directory:

```text
python3 run_compact_joined_job.py ROOT ROOT/logs/compact-joined-checks-20261013
python3 run_compact_joined_job.py ROOT ROOT/logs/compact-joined-native-20261013 --profiles
python3 run_compact_joined_job.py ROOT ROOT/logs/compact-joined-native-confirm-20261013 --profiles --confirm
python3 archive_compact_joined.py ROOT ROOT/compact-joined-evidence.tar.gz
python3 verify_compact_joined.py EXTRACTED_EVIDENCE VERIFICATION_JSON
```

All listed commands executed successfully. Do not rerun them against the
existing output/build directories. The installer hash-checks prior source bytes and preserves a
backup before installing only the isolated investigation's source files.
The original GPT-OSS and Pi checkouts remain clean.

Packing still occurs on first use; this change alone does not remove that
latency. Cross-pair reuse can retain multiple joined allocations. As with
the existing context encoding API, simultaneous calls on shared mutable
weight/context objects are outside this experiment's tested usage.
Native tests establish offset-view compatibility on the actual macOS/M4;
the API is documented in Apple's [MPSMatrix buffer/offset initializer](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix/init(buffer:offset:descriptor:)).

The evidence archive SHA-256 is
`372490609bb8656a8d041360a0d138918cb5f4ccfba6578adeb7c122fa61b307`.
The local verifier checked all 486 archived file hashes, validation stages,
dispatch counts, full outputs, chunk plans, fresh image encoding, actual
worker peak RSS and recomputed warm means/ratios. See `verification.json`
and `first-use-summary.json`. The full 2x decode and half-TTFT objective
remains unmet. The memory improvement does not establish a latency or
decode improvement, so the experiment remains default OFF.
