# Compact joined half-weight storage: validation in progress

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

## Verified so far

The isolated Mac build and CTest completed with exit 0. The native joined
tests compare complete dyadic outputs bit for bit for both admission
shapes. Added tests exercise individual projections after joining,
reversed-pair repacking and individual projections after reversing. The
existing independent vision, convolution, delta, direct-write and matrix
tests also pass. All 24 trained text exact-tail/state cases completed with
the existing numerical bounds and dispatch assertions. Coder and subsequent
numerical/cache/vision checks remain in progress; full validation is not
yet established. Pending snapshots are historical observations, not final
job results.

Candidate library SHA-256:
`b58260e7e3a8488bf888d6f3d27a957ea209b1e28f5ee1afd12cbd4dfda1762b`.

The durable validation job is PID **75480**, observed live while coder
checks were starting. Its authoritative directory is
`/Users/zsdx/garnet-metal-investigation/logs/compact-joined-checks-20261013`.
Inspect the PID and `exit.json` before advancing; an observation timeout
does not justify restarting it.

## Next required measurements

After all 18 guarded validation stages exit 0, run the original joined
library against the compact candidate sequentially in ABBA order across
all six frozen profiles, then repeat the two admitted profiles. Compare
full generated IDs, prompt IDs, finish flags and actual joined dispatch
counts. Record warm TTFT/TPS, first request after initialization, separate
initialization, and macOS time-l worker peak RSS. The comparison control is
the original joined library, SHA-256
`cc41fde1f3905fcfb02fed6a50ab0f9d68933e94358352b92db4e44cb5f43dfe`;
the validation build metadata separately fingerprints the unchanged
primary library. No fresh vLLM baseline has been run for this candidate.

Commands, from the isolated source's `test2026/metal` directory:

```text
python3 run_compact_joined_job.py ROOT ROOT/logs/compact-joined-checks-20261013
python3 run_compact_joined_job.py ROOT ROOT/logs/compact-joined-native-20261013 --profiles
python3 run_compact_joined_job.py ROOT ROOT/logs/compact-joined-native-confirm-20261013 --profiles --confirm
python3 archive_compact_joined.py ROOT ROOT/compact-joined-evidence.tar.gz
python3 verify_compact_joined.py EXTRACTED_EVIDENCE VERIFICATION_JSON
```

The first command has already executed; do not run it again against the
existing output/build directories. The remaining commands have not yet
executed. The installer hash-checks prior source bytes and preserves a
backup before installing only the isolated investigation's source files.
The original GPT-OSS and Pi checkouts remain clean.

Packing still occurs on first use; this change alone does not remove that
latency. Cross-pair reuse can retain multiple joined allocations. As with
the existing context encoding API, simultaneous calls on shared mutable
weight/context objects are outside this experiment's tested usage.
Native tests establish offset-view compatibility on the actual macOS/M4;
the API is documented in Apple's [MPSMatrix buffer/offset initializer](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix/init(buffer:offset:descriptor:)).

The full 2x decode and half-TTFT objective remains unmet. No performance or
memory benefit is claimed until the actual comparisons complete.
