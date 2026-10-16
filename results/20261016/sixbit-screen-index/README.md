# Six bit greedy head rejection index rejected by clean timing

The clean full cohort PID38347 exits0 in130.993883seconds: all original-score,
greedy, fallback and ownership checks pass, but neither unmet batch1 head gains
5% in both GPU and wall clocks. Several other profiles regress. Reject both
unchanged confirmation and native integration. The original eight-bit screening
index and all matched goal counts remain selected and unchanged.

## Mechanism and selection provenance

The [four-bit experiment](../nibble-screen-index/README.md) halves index payload
but greatly increases survivors under the original rejection bound. This new
six-bit candidate is chosen from that evidence before observing its timings.
Use per-row signed nearest-even integers in[-31,31] and exact dyadic scale,
retaining original upward error/original/reconstructed L2 metadata and the
unchanged strict-math screening guard/error allowance. It changes only the
rejection index: final exact seed/refinement use original BF16 weights and
original FP32 arithmetic, with full-logit route preserved.

Four output integers form24bits, and four K terms span three32-bit words loaded
as packed_uint3 (12bytes versus the original16). The shader reconstructs four
24-bit words, extracts six-bit tags and sign-extends exactly. Every CPU-packed
value and padded zero roundtrips before upload. Additional Windows CPU proof,
run after launch, exhausts all2^24 possible word values at each of four positions
with rotated neighboring words:67,108,864 comparisons, plus five padded/irregular
layouts. It does not exhaust all2^96 neighboring combinations or replace GPU
non-multiple-of-four vocabulary-tail tests, which remain unimplemented.

The private context retains the prototype's Nibble names for its compact-mode
flag, API and binary filenames. They identify this shared prototype mechanism;
the source shifts/masks, maximum31 and actual3/4-byte payload demonstrate that
the measured candidate is six bits. No serving source/library is changed.

## Excluded overlap and fresh clean policy

Initial PID35707 exits0 in133.932749seconds. The agent mistakenly started a large
prior four-bit archive SCP while this six-bit cohort was executing. At07:28:31UTC,
before inspecting six-bit clock ratios, save an unconditional exclusion of ALL
initial six-bit timing from selection. Its later audit/decision fields remain
raw diagnostic records; they cannot admit a repeat or model integration.
[Pre-ratio exclusion](overlap-exclusion.json),
[observed transfer terminal0](transfer-terminal-observation.json),
[excluded complete dataset](excluded-overlap/verification.json).

Only after actual SCP terminal0 and initial six-bit terminal/full archive/Mac
audit, start a fresh clean initial with identical source, defines, cases, complete
numeric gates and timing design. No transfer/archive task runs during it. This
is a prospectively recorded protocol repair, not selection of a favorable third
repeat. A qualifying clean initial could permit exactly one unchanged clean
confirmation; a failing clean initial stops. The clean initial fails, so no
confirmation is run. Resource contention was not measured as the cause of any
initial-clock value, and excluded ratios are not displayed as selection evidence.

## Clean original-score and timing evidence

Before any timing, all three models complete numeric mode. Per cohort:51captured
queries reexecute full original native logits bit-for-bit;102screening policy
checks and66unsafe checks compare47,488,512 original score values. Every retained
score bit matches, every rejected original score is below the exact seed max,
and greedy/error flags match.102ownership cases cover pending/foreign rejection,
abort/unreadability/counter clearing/recovery and uninitialized inputs. No bounds
are relaxed. The inherited rejection bounds are sampled here, not certified for
all future input magnitudes. Native session/vision/state/full-output gates remain
required before any integration.

Same timing design as four-bit:2seconds conditioning per model/batch, eight
ordinary captured steps, own-null before comparison for each query,10ABBA rounds,
3warm calls/window,5samples/window, all19,200 raw intervals retained per cohort.
Each displayed side/phase averages800intervals over the eight steps. Fractions
are candidate/control, with no filtering or null normalization. Mixed-row query
is numerical-only. Pre-execution screen: both text/coder B1 need>=5% GPU AND wall
improvement, all six groups preserve both clocks within2%, own-nulls within5%.
All clean grouped nulls pass; neither required batch1 gain passes.

| Model | Batch | GPU fraction | Wall fraction | Null GPU fraction | Null wall fraction | Mean kept rows per request row control / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 0.988159 | 1.020843 | 1.000945 | 1.000073 | 1036.125 / 2001.500 |
| Qwen3-0.6B | 2 | 1.054018 | 1.088603 | 0.999308 | 0.999543 | 1036.125 / 2001.500 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.966961 | 1.003956 | 0.997786 | 0.996599 | 1025.500 / 1037.375 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1.022544 | 1.050827 | 1.005918 | 1.005857 | 1025.500 / 1037.375 |
| Qwen3.5-0.8B | 1 | 1.028409 | 1.048038 | 0.999629 | 1.000193 | 1030.625 / 1111.750 |
| Qwen3.5-0.8B | 2 | 1.065104 | 1.085370 | 1.001295 | 1.002010 | 1030.625 / 1111.750 |

Coder B1 GPU falls3.304%, but wall rises0.396%, insufficient for its5% two-clock
screen. Text B1 GPU falls1.184% while wall rises2.084%. Text/coder B2 and both VL
profiles regress. Six bits greatly reduce four-bit survivor inflation, but this
implementation still fails the complete component requirements. No favorable
repeat, model-speed extrapolation or threshold change follows.

## Cost and archive reconstruction

Numeric-process measured index payload and upload-plus-index preparation:

| Model | Control / candidate index bytes | Control / candidate preparation ms |
| --- | ---: | ---: |
| Qwen3-0.6B | 155582464 / 116686848 | 776.604 / 854.314 |
| Qwen2.5-Coder-0.5B-Instruct | 136134656 / 102100992 | 678.990 / 743.792 |
| Qwen3.5-0.8B | 254279680 / 190709760 | 1262.572 / 1387.914 |

Candidate index payload is25% smaller. Both policy weights/indexes/metadata and
scratch remain resident together in each component process; these byte counts
are not whole-model/RSS savings. Raw0.5second RSS samples (not exact OS peak)
remain in the audits. Initialization, first-after-init, model TTFT/TPS and fresh
vLLM ratios are unmeasured. Do not combine these component clocks with older
comparator times.

Both full frozen archives remain on Mac: excluded161files SHA256
`4a2ba7e343451ca4719329948173c535889ad4411978222fddc4e360af66d0b5`;
clean160files SHA256
`eccbea1589fe2e53b9783feffe8a0aedadc8bce73c48521b3a5d34dc89633eec`.
Each contains all exact input/reference arrays,21consumed source identities, six
generated sources, defines, command/stage/guard/RSS records and omitted checkpoint,
raw head and executable identities. Mac/Windows audits reproduce identical data.
The earlier Windows POSIX-path-key compatibility repair applies unchanged.

To avoid retransmitting identical large captured arrays, Windows receives small
transports and restores102array files from the fully downloaded/hash-verified
four-bit archive. Every original manifest member is independently size/SHA
verified after reconstruction; no execution results are substituted. Windows
verifies transport gzip bytes and all original file bytes, while original full
gzip blob hashes are Mac observations, not independently Windows-verified blobs.
[Excluded transport proof](excluded-overlap/transport-verification.json),
[clean transport proof](clean-initial/transport-verification.json).

Transport SHA256s: excluded
`4b0897ba6d5a4c91ab3b8464594eba6e73bb35f7054c1c8d023d1515589a222d`;
clean `a6b0dad9b711a6b17ec203f5d13fcf8cb136390b0ea472d1530324c3ef07d97d`.
All selected execution sources are immutable across excluded/clean runs. The
external portable verifier and transport tools are provenance-only code, created
separately from the consumed runner/generator/harness. Environment is captured
per cohort: M4/16GiB, macOS15.6.1, clang17, Python3.14.6; no cache flush or thermal
causality measurement. All three controllers are terminal and archived.

```text
python -X utf8 test2026/metal/prepare_sixbit_screen_index.py REPOSITORY FRESH_OUTPUT
python3 launch_sixbit_screen_index.py ROOT ROOT/logs/sixbit-screen-index-clean-initial-20261016
python3 archive_sixbit_screen_index.py ROOT ROOT/logs/sixbit-screen-index-clean-initial-20261016 ARCHIVE
python3 transport_screen_index_evidence.py FULL_ARCHIVE FRESH_TRANSPORT
python -X utf8 test2026/metal/restore_screen_index_evidence.py TRANSPORT TRANSPORT_SHA FOUR_BIT_RESULT FRESH_OUTPUT
python -X utf8 test2026/metal/verify_sixbit_screen_index.py FRESH_OUTPUT AUDIT_JSON
```

[Clean Windows audit](clean-initial/verification.json),
[clean Mac audit](clean-initial/mac-verification.json),
[clean complete source/data manifest](clean-initial/evidence/manifest.json),
[pre-launch CPU source preparation](preparation/preparation.json),
[post-launch exhaustive CPU validation](exhaustive-preparation/preparation.json).
Reference library `d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.
