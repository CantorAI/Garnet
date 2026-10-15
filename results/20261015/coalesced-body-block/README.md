# Exact coalesced BF16 block storage: repeated component signal

Compact blocks contain four output channels by128 K terms. Each stores512
sign/mantissa bytes and256 four-bit exponent-tag bytes plus a32-bit exponent
base/compact flag. When the exponent range exceeds15, the entire SIMD group
uses the original four-output raw BF16 buffer. Compact payload has fixed pitch
and coalesced vector loads. There are no per-value escapes or prefix sums.
Original BF16 bits are reconstructed before the original FP32 arithmetic and
SIMD reduction. This is lossless storage, not a reduced-precision model.

This differs from the earlier one-output nibble/whole-block candidates in its
four-output vector loading, fixed byte-plane addresses and uniform block
fallback. The original master and raw four-output storage remain resident;
the experiment does not establish process-memory savings. Added payload and
CPU packing costs are outside component timing and must be measured at model
initialization before any deployment decision.

Independent CPU admission checks original checkpoint slices and reports20
eligible first-layer matrices:6 text,3 coder and11 VL. Approximately96.5% of
their blocks fit the compact payload. This is not full-model compression
coverage. The CPU report/source are frozen alongside GPU evidence; the native
context additionally packs real byte/nibble planes and checks every recovered
original weight bit before uploading any compact buffer.

PID87646 exited0 in20.5991s. After independent audit, confirmation PID88145
exited0 in20.5885s. Combined independent verification checks89 archived files,
144 numerical cases,378 safety profiles and2304 timed submissions. All
candidate outputs are bit-exact against the current native four-output body
and meet the unchanged FP64 envelope. Runtime packing counts match the
independent CPU report. All original matrices, unsupported routes and the
special float-key policy remain in the workload.

The selected64-thread policy is fixed across all model/batch profiles:

| Model / batch | Initial GPU / wall fraction | Repeat GPU / wall fraction |
| --- | ---: | ---: |
| Text / 1 | 0.8644 / 0.9228 | 0.8361 / 0.9131 |
| Text / 2 | 0.8609 / 0.9157 | 0.8320 / 0.9070 |
| Coder / 1 | 0.8379 / 0.9107 | 0.8356 / 0.8797 |
| Coder / 2 | 0.8464 / 0.8811 | 0.8014 / 0.8632 |
| VL / 1 | 0.9660 / 0.9811 | 0.9552 / 0.9817 |
| VL / 2 | 0.8112 / 0.8577 | 0.8271 / 0.8966 |

Fractions are candidate/control; below1 is faster. All64/128/256-thread raw
windows remain available. The common64-thread choice improves all six groups
in both trials, avoiding a per-model dispatch policy selected from individual
best samples. Some other variants regress. Every profile uses two ABBA rounds
and eight samples per position, after common warmup. Resource guards exclude
competing inference. These are independent first-layer projections streamed
in one submission, not complete-model execution or a vLLM comparison.

A native OFF-default integration is warranted for full validation. Whole-model
outputs, decode throughput, TTFT, initialization and memory costs remain
unproven for this route. The full2x decode / half-TTFT goal remains unmet.

Initial archive SHA256:
`de81ef6af54996b052999986fa8669c29b1d74a23883b5348770967fd8241f1f`.
Combined archive SHA256:
`21506da73db15d88928873e0cb7effe557d9d28bbfeb5d015c8e526cd936a24a`.

```text
python3 audit_coalesced_body_block.py ROOT ROOT/logs/coalesced-body-block-admission-20261015
python3 launch_coalesced_body_block.py ROOT ROOT/logs/coalesced-body-block-20261015 checks
python3 archive_coalesced_body_block.py ROOT ARCHIVE
python -X utf8 verify_coalesced_body_block.py RESULTS VERIFICATION_JSON
python3 launch_confirm_coalesced_body_block.py ROOT ROOT/logs/coalesced-body-block-confirm-20261015 checks
python3 archive_coalesced_body_block.py ROOT ARCHIVE coalesced-body-block-20261015 coalesced-body-block-confirm-20261015
```
