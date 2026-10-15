# Retuned native VL B2 prefill on the same library

The 64-token internal maximum gives a small repeated warm-TTFT reduction;
16 tokens is slower and rejected. This changes the internal native schedule
while preserving all 79 logical prompt tokens, 80 total physical positions,
batch 2, fresh 288x192 image encoding, checkpoint values, precision, capacity
256, output cap 32, stops and complete outputs. It is a separate native study,
not an update to frozen-profile native/vLLM ratios. The full goal remains unmet.

Both timed policies use exactly the same native library, SHA-256
`837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471`.
The library's original 27 stages and 144 complete outputs already passed
in [the layout study](../interleaved-head-native/README.md). Both policies
keep the same interleaved screening index, exact 1024 seed, streaming,
partial reset, serial vision prefetch and combined threaded/striped frontend.
The only case-field difference is `chunk`. The actual plans are control
`[32,32,16]`, rejected candidate `[16,16,16,16,16]`, and candidate `[64,16]`.

Before timing, independent NumPy references exercise first/last trained
linear and full-attention layers at chunks 16/32/64 and batch 2. Twelve
profiles and 36 carried-state steps pass the unchanged
`5e-6 + 5e-5*abs(reference)` envelope. Worst normalized error is
0.6186854863, below 1. Explicit and resident outputs are exactly equal.
Inactive tokens, overwritten slots, invalid controls, rejected-state
preservation, reset and closed-graph rejection checks also pass.

All passing fixtures are saved, including their input/expected-array sizes
and hashes. Raw float arrays remain on the Mac and are omitted from Git.
The 64-token repeat reuses these immutable passed references only after
source, checkpoint/library/runtime identities and the first audit are checked.
It still uses new sequential serving workers and freshly encodes every image.
No numerical allowances or logical workloads were changed to obtain a gain.

Initial PID 63436 exited 0 in 244.2474 s, including the reference gate and
both profile comparisons. Repeat PID 68104 exited 0 in 38.6424 s. The audit
checks 406 archived files, all reference/profile shapes, original-bound logs,
36 complete requests and recomputed raw means. Each comparison uses fresh
control/candidate/candidate/control workers with three requests each. Four
warm requests per policy exclude first use after initialization. Inference
runtimes do not intentionally overlap, and guards sample every 0.5 s.

| Internal maximum / trial | TTFT control / candidate ms | TTFT change | Decode control / candidate tokens/s per row |
| --- | ---: | ---: | ---: |
| 16 / initial | 321.256 / 355.937 | +10.80% | 50.098 / 49.928 |
| 64 / initial | 320.642 / 314.553 | -1.90% | 50.137 / 50.120 |
| 64 / repeat | 319.148 / 309.454 | -3.04% | 49.919 / 50.259 |

The 64-token plan reduces actual streaming partial submissions from 15 to 10
per prefill; the 16-token plan uses 25. These are implementation differences,
not isolated attribution of the entire latency change. Decode is effectively
flat: -0.03% / +0.68% for the 64-token candidate. The candidate is retained
for subsequent latency investigation; it does not silently replace the
frozen benchmark plan or establish a new comparator advantage.

Initialization, first request after initialization, separate image/prefill
clocks and peak individual child RSS are retained in raw worker records.
Caches were not flushed. Overlapping clocks must not be summed, and peak
RSS is not unique Metal memory. No observed hardware-cold end-to-end claim
is made by combining separate clocks.

The current decoder projection admission is at most 128 rows. Batch 2 with
80 prompt positions would require 160 rows and is not currently admitted;
there is no result for that profile here. A separate bounded implementation
and original-envelope numerical/serving validation would be required before
testing it. Current 16/32/64 plans all stay within the existing 128-row guard.

Initial archive SHA-256:
`c1e015b90e8c94fb84a18c2027bf989b546892815b2f6009a8730ee2f1005cae`.
Combined archive SHA-256:
`8ce7bd7387e3c085524260ae7a62c15f8f7a50dc142ba5088365cd3b4bca940f`.

```text
python3 launch_replanned_vl_prefill.py ROOT ROOT/logs/replanned-vl-prefill-20261015 checks
python3 archive_replanned_vl_prefill.py ROOT ARCHIVE
python -X utf8 verify_replanned_vl_prefill.py RESULTS VERIFICATION_JSON
python3 launch_confirm_replanned_vl_prefill.py ROOT ROOT/logs/replanned-vl-prefill-confirm-20261015 checks
python3 archive_replanned_vl_prefill.py ROOT ARCHIVE replanned-vl-prefill-20261015 replanned-vl-prefill-confirm-20261015
python -X utf8 verify_replanned_vl_prefill.py RESULTS VERIFICATION_JSON
```
