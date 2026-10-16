# Four bit greedy head rejection index rejected

Actual Mac PID32841 completed in144.568587seconds, exit0. All original captured
score, exact-greedy, unsafe fallback and ownership checks pass; all six timing
profiles regress. GPU fractions1.393651–1.600489 and wall1.340444–1.563867 reject
confirmation/integration. All six grouped own-nulls are within5%. This is a
private vocabulary-head component; the matched three-model target is unchanged.

## Hypothesis and exact original control

The retained head uses a derived interleaved INT8 row index to reject vocabulary
rows before original BF16/FP32 refinement. Reduce only that index to four signed
bits, halving its payload. Row quantization remains nearest-even with an exact
dyadic scale; the candidate changes its maximum magnitude from127 to7. Keep the
original upward residual/original/reconstructed L2 metadata and strict-math
error allowance, query norm/magnitude guard, exact first1,024-row seed, original
BF16 weights, refinement order and final argmax. This index is used only for
rejection; learned final logits do not use four-bit model weights.

Four output tags fit in a16-bit word; four K terms load8contiguous bytes per lane,
versus16bytes in the retained interleaved index. Signed nibble unpacking recovers
the approximate integer exactly. Every actual CPU-packed integer and padded zero
is roundtripped before upload. The original INT8 control is generated privately
with its unchanged default maximum127 and original shader. Both arms use the
accepted d52 context, identical defines and strict screening compiler options;
the selected serving library is not modified. The component holds both original
weight/index policies concurrently in memory but executes them sequentially.

## Complete original-score and safety checks

All three models finish numeric mode before any timing.51actual captured queries
(17/model:8ordinary B1 steps,8B2 steps and one mixed-row fixture) reexecute the
original full BF16 head and match every frozen GPU score bit. Both screening
policies check every retained score bit and every rejected original score below
the actual exact-seed threshold, plus original greedy token/error flag.102policy
query checks and66zero/tiny/large/NaN/Inf/mixed unsafe checks compare47,488,512
original score values.102ownership cases check pending/foreign writer rejection,
abort counter clearing/unreadability, successful recovery and uninitialized input.

Unsafe rows retain the existing full original refinement. These sampled learned
queries do not certify the inherited screening bounds for arbitrary future
inputs. No numerical tolerance or original predicate was weakened. GPU fixtures
with non-multiple-of-four vocabulary tails, native session/vision/carried-state
validation and full inference remain required before any integration. Windows
CPU preparation covers256signed byte patterns and five padded/irregular layouts;
CPU layout tests are not GPU tail coverage.

## Timings and survivor inflation

Two seconds alternating conditioning precedes step0 at each model/batch. For
each of eight ordinary steps, own-null control/control runs before comparison,
10ABBA rounds per phase,3warm calls/window,5samples/window. All19,200 measured
intervals retained; each grouped side/clock/phase averages800samples over eight
steps, with no filtering, null normalization or selected subset. The mixed-row
fixture is numeric-only. Fractions are candidate/control; above1 is slower.

Pre-execution admission requires both unmet text/coder B1 heads improve GPU and
wall by5%, every group preserve both clocks within2%, and all six grouped
own-nulls stay within5%. A qualifying initial permits one complete unchanged
confirmation; an initial failure stops. Neither confirmation nor integration
is admitted. Grouped nulls are the declared screen; no per-query statistical
significance or population confidence interval is claimed.

| Model | Batch | GPU fraction | Wall fraction | Null GPU fraction | Null wall fraction | Mean kept rows per request row control / candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 1.588598 | 1.528325 | 0.997524 | 0.997283 | 1036.125 / 77755.875 |
| Qwen3-0.6B | 2 | 1.593453 | 1.541265 | 0.999789 | 1.001314 | 1036.125 / 77755.875 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 1.393651 | 1.340444 | 1.000233 | 1.001578 | 1025.500 / 62643.750 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 1.413873 | 1.362159 | 1.000587 | 1.003011 | 1025.500 / 62643.750 |
| Qwen3.5-0.8B | 1 | 1.593808 | 1.555631 | 0.997776 | 0.998196 | 1030.625 / 126722.500 |
| Qwen3.5-0.8B | 2 | 1.600489 | 1.563867 | 1.000428 | 1.000636 | 1030.625 / 126722.500 |

The four-bit index retains far more rows under the original, now wider residual
bounds. These actual counts document reduced pruning; they do not isolate the
relative contributions of extra original-weight reads, unpacking and scheduling
to total GPU time. Packing alone is not a model-speed result. Do not repeat the
rejected candidate unchanged or substitute a smaller bound to reduce survivors.

## Costs and reproducibility

For text/coder/VL the INT8 index payloads are155,582,464 /136,134,656 /254,279,680
bytes; four-bit payloads are77,791,232 /68,067,328 /127,139,840bytes. Both indexes,
original BF16 copies, metadata and scratch coexist in this component. These
payload savings are not process-RSS or model-memory savings. Numeric preparation
records control/candidate upload-plus-index times779.773/818.935,680.456/717.124,
1,266.088/1,339.246ms.0.5second RSS samples (not OS peak accounting) record about
1.1–1.9GiB across separate numeric/timing processes; raw per-process maxima remain
in the audit. No full-model initialization, first-after-init or TTFT/TPS measured.

The159-file terminal archive includes six generated sources,20consumed source
identities, defines, stage/guard/RSS/exit records, all captured input/full-reference
arrays, commands and size/hash identities of omitted checkpoints/raw heads and
executable. Both-host audits reproduce identical decisions. An initial Windows
verifier failed while converting an archived POSIX path through Windows Path;
only manifest-key normalization was repaired, without GPU/data/gate changes.
The original Mac verifier and its success remain frozen. Raw native compilation
uses macOS15.6.1, M4/16GiB, clang17 and Python3.14.6 as captured in metadata.

```text
python -X utf8 test2026/metal/prepare_nibble_screen_index.py REPOSITORY FRESH_PREPARATION
python3 launch_nibble_screen_index.py ROOT ROOT/logs/nibble-screen-index-20261016
python3 archive_nibble_screen_index.py ROOT ROOT/logs/nibble-screen-index-20261016 ARCHIVE
python -X utf8 test2026/metal/verify_nibble_screen_index.py RESULTS/initial VERIFICATION_JSON
```

[Windows audit](initial/verification.json), [Mac audit](initial/mac-verification.json),
[Manifest and raw records](initial/evidence/manifest.json),
[Full archive byte verification](initial/archive-verification.json),
[CPU preparation](preparation/preparation.json),
[Audit compatibility repair](audit-compatibility-repair.json).

Archive SHA256 `de0ccb3e0085a73793d839b0e8d403ab175ea538ee13dd1e4b64a9fc249e8a85`.
Reference library `d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e`.
No other inference/probe ran during the measured component. The large terminal
archive transfer occurred after this four-bit run, overlapped a subsequent six-bit
cohort, and was independently SHA/manifest verified only after actual terminal0.
That subsequent cohort is excluded separately; this four-bit cohort is retained
as the clean rejected result. No hardware cache flush or thermal attribution.
