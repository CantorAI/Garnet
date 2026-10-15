# Padded-row native MPS body projection: rejected

Actual owned PID 59445 exited 0 after 34.2630 s. The experiment invokes the
existing Garnet `MetalContext` through its selected static Metal library.
Original B1/B2 inputs are padded to physical rows 4/8/16 to exercise the
existing mixed-storage MPS matrix path. Original BF16 weights, their lossless
common-scale half representation, FP32 input/output and original raw BF16
decode control are retained. No serving route or production source is changed.

Independent verification covers 24 archived files, 648 numerical checks,
162 admitted matrix/batch/padding profiles and 10,368 timed submissions.
All sampled numerical checks pass unchanged bounds: the FP64 forward-error
envelope and `5e-6+5e-5*abs(original)` comparison. All extra padded output rows
remain zero and every candidate reports an actual common-scale MPS dispatch.
MPS reduction order can change FP32 result bits; this is bounded numerical
agreement, not a bit-exact projection claim.

Twenty-nine original trained body matrices are available: seven each for text
and coder, fifteen across VL linear/full-attention layers. Two VL matrices do
not admit the lossless common-scale half representation and are excluded.
The runner verifies every raw matrix against its unchanged checkpoint bytes.
Four deterministic input patterns include zero, alternating signs and a small
scaled input. Every admitted profile has two ABBA rounds, two untimed warmups
and eight timed operations per position, after 250 ms common warmup.

| Model / batch | Sum GPU fraction, pad 4 / 8 / 16 | Sum wall fraction, pad 4 / 8 / 16 |
| --- | ---: | ---: |
| Text / 1 | 1.258 / 2.179 / 1.185 | 1.152 / 1.411 / 1.156 |
| Text / 2 | 1.118 / 2.093 / 1.173 | 1.119 / 1.428 / 1.199 |
| Coder / 1 | 1.114 / 2.233 / 1.401 | 1.163 / 1.415 / 1.273 |
| Coder / 2 | 1.230 / 2.407 / 1.466 | 1.158 / 1.431 / 1.291 |
| VL / 1 | 1.250 / 2.059 / 1.230 | 1.183 / 1.350 / 1.181 |
| VL / 2 | 1.140 / 2.092 / 1.309 | 1.126 / 1.414 / 1.228 |

Fractions divide the sums of measured candidate component means by the sums
of raw component means. Every aggregate is slower. These are repeatedly
resident individual matrices; initialization, allocation, padding/copy and
validation/readback are outside timing. They are not streamed model timing,
decode TPS, TTFT, memory-bandwidth observations or a vLLM result. No favorable
repeat or integration is warranted. All runtimes/probes ran sequentially with
0.5 s guards. The full goal remains unmet.

Archive SHA-256:
`a6d6956488dc009aa91d09dc044ca6a173e6e9c9d8ff07215f99fef4156db210`.
Commands, learned-byte/source/binary hashes, compiler/hardware versions,
numerical records and all raw timing windows are in `evidence/`.

```text
python3 launch_padded_mps_body.py ROOT ROOT/logs/padded-mps-body-20261015 checks
python3 archive_padded_mps_body.py ROOT ARCHIVE
python -X utf8 verify_padded_mps_body.py RESULTS VERIFICATION_JSON
```
