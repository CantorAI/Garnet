# Lossless four-column screening index: repeated component gain

Two actual same-process trials reduce component GPU and wall time while
preserving every sampled masked score and flag bit. Four output columns are
stored together so the shader uses one uint4 vector load for their four-key
tile instead of four separate char4 loads. Index values, FP32 FMA order,
SIMD reductions, bounds, seed and original BF16/FP32 refinement are unchanged.
This is a derived screening-index layout change, not model quantization.

Both policies use an exact 1024-row seed. Before timing, the raw original
head must match captured complete logits bit for bit. Each policy checks all
retained scores, every sampled rejection against full original scores, and
greedy output. Candidate and control additionally match all masked output
bits and both visit/keep flag arrays. The existing candidate unsafe-input
tests exercise zero/tiny/large/NaN/Inf/mixed rows and full-refinement fallback.
No bounds were widened. The sampled gates are not certification for all
future inputs.

The complete byte permutation roundtrips all 155,582,464 / 136,134,656 /
254,279,680 derived index values for text, coder and VL, respectively, in
each trial. Physical row count rounds up to a four-column group; padded
entries are zero. The original index byte hashes and all learned input/head
hashes are verified before use. All original BF16 weights remain authoritative.

Initial PID 30117 exited 0 in 38.2029 s. Independent repeat PID 30957 exited
0 in 38.1735 s. The audit checks 84 archived files, six complete index
roundtrips, 102 captured-query pairs, 156 row states per policy, 66 candidate
unsafe-input cases, 6,528 raw timed executions, guards and ABBA ordering.
Each query has at least 250 ms common warming, then two ABBA rounds with
eight samples per window. Timed intervals include exact seed, bounds,
screen/refine and argmax in one command buffer. Only one probe runs at a time.

Fractions below average ordinary captured steps and exclude the separate
mixed-row fixture. Candidate/control below 1 means faster.

| Model / batch | GPU fraction initial / repeat | Wall fraction initial / repeat |
| --- | ---: | ---: |
| Qwen3-0.6B / 1 | 0.883285 / 0.875999 | 0.894012 / 0.891909 |
| Qwen3-0.6B / 2 | 0.848307 / 0.849839 | 0.844705 / 0.838558 |
| Qwen2.5-Coder-0.5B-Instruct / 1 | 0.885125 / 0.889093 | 0.893861 / 0.892792 |
| Qwen2.5-Coder-0.5B-Instruct / 2 | 0.853382 / 0.853132 | 0.851345 / 0.857764 |
| Qwen3.5-0.8B VL / 1 | 0.844977 / 0.842560 | 0.847968 / 0.846390 |
| Qwen3.5-0.8B VL / 2 | 0.834649 / 0.832554 | 0.828492 / 0.825695 |

Wall reductions repeat at 10.6–10.8% for text/coder B1, 14.2–16.1% for
text/coder B2, 15.2–15.4% for VL B1 and 17.2–17.4% for VL B2. These are
component improvements, not decoder tokens/s or TTFT gains. No native
integration or new vLLM comparison has been executed for this layout yet.
The full goal remains unmet.

The next native experiment should add an OFF-by-default layout option,
retain the original shape/admission and fallback rules, verify every packed
index value before publication and pad the final four-column group. The
Metal macro-selected body must match this audited component. Actual layout
dispatch counters should reset/publish through context and session paths;
workers should check every prediction against the selected policy. Synthetic
nonmultiple-of-four heads, outside-seed ties, trained heads, generic logits,
poison/reset/close/rebuild and private-session checks must pass before full
sequential native ABBA and a fresh repeat. The 1024-versus-8192 tradeoffs
remain separate in [the small-seed native record](../small-seed-native/README.md).

Both Objective-C++ harnesses compiled on the M4. Xcode's offline Metal tool
is a stub without its optional Metal Toolchain, so it could not compile a
standalone shader. The actual runtime Metal compiler subsequently compiled
and executed both complete trials successfully. No toolchain was installed.

Initial archive SHA-256:
`7534b3f700cd246b61a6effc28fa84f2565853ef3b75e38364bc70d4a363d7c6`.
Combined archive SHA-256:
`eafb2ad9a499d2c605db92685ee719284d9be599cfd8280e96bb53b1cfbbb79d`.

```text
python3 launch_interleaved_screen_index.py ROOT ROOT/logs/interleaved-screen-index-20261015 checks
python3 launch_interleaved_screen_index.py ROOT ROOT/logs/interleaved-screen-index-confirm-20261015 checks
python3 archive_interleaved_screen_index.py ROOT ARCHIVE interleaved-screen-index-20261015 interleaved-screen-index-confirm-20261015
python -X utf8 verify_interleaved_screen_index.py RESULTS VERIFICATION_JSON
```
