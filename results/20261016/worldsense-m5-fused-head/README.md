# WS-M5-01: fused greedy head through WorldSense — rejected screen

This is an actual WorldSense/XLang3 → Galaxy Garnet-VLM → native Garnet Metal
experiment on the MacBook M5/24 GiB. It is a deployment-specific evaluation of
an existing optional kernel, not a claim of a new fused-head algorithm. The
candidate changes only `GARNET_METAL_FUSED_HEAD_ARGMAX=OFF→ON` relative to the
deployed seven-flag BF16/FP32 Qwen3.5-4B profile. No CUDA/default change,
quantization, reduced image budget, token cap or cached model answer is used.

## Protocol and scope

`protocol.json` records the predeclared screen and machine/base commits;
`cohort-manifest.json` records consumed libraries/serving hashes, process IDs,
system VM/swap and post-run RSS. The four fresh processes run sequentially
control/candidate/candidate/control. There are three immutable archived JPEG
frames from pedestrians, bunny and car videos. Skill cases are an absent
elephant, search-only description, and current-image vehicle presence. The
normal Galaxy filter also calls the 1.7B text reasoner for current-event
verification; a fixed text search-planning request exercises the LLM route.

WorldSense temporarily clears native live Skill bindings during each suite,
restores bindings from the authoritative DB in `finally`, and returns results
without event delivery, WorldState commits or result-store writes. The normal
current-Skill inference and search-planning implementations are reused. This
excludes video decoding, candidate scheduling, temporal memory, deduplication,
event persistence and search retrieval/ranking. The separate live-video
baseline is integration evidence, not matched speed evidence.

Each worker first warms all three frames and the search request, then repeats
the identical sequence twice. All first/warmup calls remain in raw JSON;
only the two later repeats enter the warm medians. There are 18 complete model
calls per worker (72 total), including 12 warm calls per worker. Frame/Skill
hashes, prompt lengths, chunk plans, full generated token IDs and texts are
saved. Each timing worker uses XLang3; a local credential provider and OS
lifecycle controller use CPython, outside inference and measured model calls.
No standalone Garnet inference timing establishes the decision.

The screen requires exact complete outputs, ≥5% 4B decode improvement and ≤2%
median TTFT/total regression in each comparison; the 1.7B route must preserve
decode within 2%. `analyze.py` independently computes these from full records.
No confidence interval, significance, whole-system FPS or general speedup is
claimed. This is a small fixed-case screen, not a publication confirmation.

## Results

| Process | 4B total p50 (s) | 4B TTFT p50 (s) | 4B decode p50 (tokens/s) |
|---|---:|---:|---:|
| control-1 | 16.762 | 6.100 | 10.950 |
| candidate-1 | 14.587 | 5.150 | 12.610 |
| candidate-2 | 14.907 | 5.340 | 12.518 |
| control-2 | 14.466 | 5.213 | 12.596 |

All complete token sequences, texts, frame/Skill identities and input counts
match across the four workers. Actual fused decode counters are nonzero only
for the candidate. The apparent first-pair improvement does not survive the
second control: 4B total rises 3.05%, TTFT rises 2.42%, and decode falls 0.62%.
The second 1.7B comparison has 2.69% higher total and 9.58% higher TTFT, despite
1.99% higher estimated decode. The declared gates therefore fail. **Reject
promotion; original deployed profile restored.** No unchanged favorable retry
is used to overturn the result. The control drift is retained, not discarded;
these observations do not isolate its cause or prove the fusion itself caused
all latency differences.

Decode is estimated as `(generated_tokens−1)/(total_seconds−ttft_seconds)`;
the generated EOS prediction may add work, so this is not a hardware-only
kernel rate. Model timings exclude loading, lock queueing, retries and storage.
Native-call and HTTP wall times are separate. GPU/encode/wait counters can
overlap and must not be added as independent costs. Post-run process RSS is
not peak memory or total unified GPU memory; VM/swap snapshots are system-wide.

The first control's warm 4B stage medians are reset 0.018s, host inputs 0.002s,
vision 0.351s, prefill 5.683s, decode 10.469s. These nonadditive medians motivate
a distinct prefill candidate; they do not justify changing the failed head
screen's gates. Telemetry retains token IDs and per-step counters, adding CPU
and diagnostic-storage overhead equally to baseline and candidate.

## Failures and correctness

An initial detached launcher produced no timings and was replaced by an
attached exec session. A native unit test initially misclassified normal
current-image text verification as temporal reasoning; the corrected assertion
requires no temporal reads or state writes. The first HTTP smoke failed to
marshal an unset native `state_changes` value. Initializing an empty list fixes
the transport; the failed response and original logs remain archived. No samples
were returned from that failed smoke; neither repair changed numerical gates.

Native structured-response and state-observability tests pass, including
current-frame admission and serializable no-write state changes. Native
matmul and fused-head bit/FP64/tie/nonfinite/fallback/recovery tests pass at
the original shapes. Extended 2048/2560-width tests are run separately and
their terminal results must be consulted; they cannot retroactively turn this
failed performance screen into an accepted optimization. XLang3 admission
tests reject temporal/geometry cases; unauthenticated benchmark HTTP returns
401. The new recorded benchmark API is opt-in and disabled without the local
`WORLDSENSE_ENABLE_BENCHMARK=1` environment setting.

Raw JSON, lifecycle logs, build/test failures, scripts, base commits, patches
and initial file identities are retained here. Model weights are external
official checkpoints; this archive is not a complete standalone distribution
of WorldSense, video licenses or trained model assets.
