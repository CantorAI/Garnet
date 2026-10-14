# Compute-pass GPU timestamps: validated diagnostic, substantial overhead

This adds evidence for choosing the next optimization; it establishes no new
serving speed gain. Production sources and selected libraries are unchanged.
The original source SHA-256 is
`b0cf78db529b4837022a66ac06db879774e114a73285d43065582124b505f480`.
One generated diagnostic source instruments twelve encoder sites in a
separate source/build tree, with the selected global-alpha options and both
dirty-prefix reset and streaming off. Independent verification proves all
raw Metal shader payloads unchanged.

The actual Apple M4 supports stage-boundary timestamps, but does not support
dispatch- or blit-boundary sampling. Capability PID **56549** exited 0 after
2.36 seconds. Two dependent compute passes produced 4096 bit-exact values and
four nonzero, within-pass monotonic timestamps in one command buffer. Apple's
[sampling documentation](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)
requires checking supported sampling boundaries. Its
[counter documentation](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
describes resolving counter buffers and clock conversion; recorded raw ticks
are not assumed to be milliseconds.

Initial diagnostic PID **57117** exited 1 after 21.27 seconds. Generation,
configure, build, numeric CTest and the first two native workers exited 0,
but trace stderr interleaved inside result stdout JSON. The strict parser
rejected it. `failed/` and `failure-verification.json` preserve 202 verified
files, the successful capability/CTest and three complete quiet outputs.
Traced outputs from that failed run remain unverified. Failure archive SHA-256:
`c907051262a914a881f254e786ff85ecb8d3361b729922ed2ebbc74c5e2f4b37`.

The repair writes traces to an owned, separate file with a process-wide mutex
and explicit flush, closes streams at context destruction, and leaves old
source/build artifacts intact. No parser tolerance was introduced. Repaired
PID **58044** exited 0 after 133.99 seconds. All six model/batch profiles ran
sequential quiet/traced/traced/quiet fresh workers, each with three original
requests. Independent `verification.json` checks 434 archived files, all 28
terminal stages, all 72 complete token/prompt/finished outputs and fresh frame
hashes, 358872 counter samples and 660 command summaries. Sample indices are
contiguous and timestamps valid. Counter records explicitly reject serialized
operation profiling and partial submissions. Checkpoint/config/tokenizer
hashes match the frozen comparison assets.

Successful archive SHA-256:
`0180f78d95067cef01bec0c463514962565cd8b83086a072727bbb02ae67c978`.
The generated source SHA-256 is
`f63adc3aaf44e3dd8c13f366ed2e33b8d7d1462b7f75d90e6057e2cc6f8315ec`.
The exact compiled library identity is archived in metadata and manifest.

`shape-ranking.json` groups observed decode-width head commands by operation
and shape. Text/coder MLP gate/up projections lead; VL's vocabulary head leads,
followed by MLP gate/up and QKV projections. BF16 projection passes account
for 79–84% of the summed sampled intervals, and heads 18–25%. These are not
percentages of full serving time: intervals overlap, internal MPS passes are
unmeasured, and the selection uses observed shapes rather than request/phase
markers. The report explicitly permits final prefill heads and small physical
tails and includes first and warm requests. Never add the sample durations
to infer command wall time or claim isolated warm throughput.

Instrumentation materially changes end-to-end clocks. Four warm requests per
mode/profile give quiet → traced decode rates of text B1 59.50→37.48,
text B2 57.24→40.69, coder B1 71.78→39.14, coder B2 70.40→41.72,
VL B1 47.61→34.29 and VL B2 44.18→31.97 tokens/s per row. This is diagnostic
overhead, not a regression of the selected production binary or a vLLM
comparison. Sparse sampling is required before using absolute timing estimates
to guide a new production candidate. The full 2x decode / half-TTFT goal
remains active and unmet.

```text
python3 launch_metal_counter_capabilities.py ROOT OUTPUT
python3 archive_counter_trace_failure.py ROOT ARCHIVE
python verify_counter_trace_failure.py FAILED_EVIDENCE VERIFICATION_JSON
python3 launch_counter_trace_profiles.py ROOT OUTPUT
python3 archive_counter_trace_profiles.py ROOT ARCHIVE
python verify_counter_trace_profiles.py EVIDENCE VERIFICATION_JSON
python summarize_counter_trace_profiles.py EVIDENCE SHAPE_RANKING_JSON
```
