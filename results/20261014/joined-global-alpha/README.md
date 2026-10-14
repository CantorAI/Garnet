# One joined common-alpha MPS product with output views

This is a component experiment, not an integrated inference gain. Both actual
M4 trials exit 0. All **20,848 validated executions** across 78 checked profiles
have zero numerical difference from the two-product control and pass the
unchanged independent FP64 envelope. Worst normalized FP64 error is
0.226054538 under `5e-6 + 5e-5*abs(reference)`, below 1. This is numerical
equality, not a signed-zero bit-identity claim.

The candidate concatenates trained gate/up BF16 matrices, encodes one common
lossless FP16 scale, and writes one MPS product directly into the joined FP32
output. Two output subviews expose each half with the correct shared row stride.
The control executes two MPS products into the same output subviews, using the
same lossless encoded weights, common inverse scale and FP32 inputs. Neither
route uses a rescale or split compute encoder. Setup/allocation/uploads and
validation remain outside the recorded clocks. Internal MPS kernel counts are
not claimed. Learned-value restoration is checked bitwise before running.

The three trained layer-0 pairs come from the frozen Qwen3-0.6B,
Qwen2.5-Coder-0.5B-Instruct and Qwen3.5-0.8B checkpoints. Canonical hashes are
checked against actual safetensors payloads. Each model uses all 13 row counts
8/16/22/24/26/32/44/48/52/64/80/128/160, four fresh power-of-two input variants,
full output checks and independent FP64 references. Every profile alternates
common warming for at least 250 ms, then two ABBA rounds, four windows/round
and twelve samples/window. Each run has 3,744 timed executions. There are no
excluded common-scale pairs.

| Model / rows | Initial / repeat encode fraction | Initial / repeat wall fraction | Initial / repeat GPU fraction |
| --- | ---: | ---: | ---: |
| Text / 24 | 0.5531 / 0.5660 | 0.9413 / 0.9352 | 0.9309 / 0.9355 |
| Text / 44 | 0.5544 / 0.5430 | 0.9050 / 0.8820 | 0.8643 / 0.8358 |
| Coder / 26 | 0.5571 / 0.5501 | 0.9367 / 0.8661 | 0.8961 / 0.8152 |
| Coder / 52 | 0.5870 / 0.5855 | 1.0917 / 0.9738 | 1.1642 / 0.9212 |
| VL / 32 | 0.5497 / 0.5742 | 0.8199 / 0.9201 | 0.8110 / 0.8591 |
| VL / 64 | 0.5518 / 0.5530 | 0.8724 / 0.9971 | 0.8987 / 0.9879 |
| VL / 80 | 0.5593 / 0.5693 | 1.0057 / 0.8929 | 1.0259 / 0.8713 |

Fractions are candidate/control. Encode duration decreases consistently, but
GPU/wall behavior depends on shape. Text 24/44, coder 26 and VL 32 show repeated
component directions. Coder 52 and VL 80 regress initially; VL 64's repeat wall
gain nearly disappears. No broad admission or whole-model gain follows from
this table. The full row-count evidence is retained in `summary.json`.

A native graph experiment must preserve output ownership, strides, aliases,
pending-state validity and lifetime across graph reuse. The component's shared
views must not be substituted into production merely from projection parity.
Remaining integration requires full model/cache/vision/prefill validation,
complete-output tests and fresh guarded serving measurements. No production
route or selected library is changed here. The 2x decode / half-TTFT goal
remains active and unmet.

Initial PID 50341 exits 0 in 47.24 seconds; fresh confirmation PID 51303 exits
0 in 46.38 seconds. Both start after the prior owned inference job exits.
The strict guard samples every 0.5 seconds and terminates only owned processes
on conflicting inference/probes. OS caches and GPU frequency are uncontrolled;
shorter external activity between samples cannot be categorically excluded.

Initial archive SHA-256
`90b268db963feea70edb50cae8f904b3dfc3303239dca0e2c2f7d0413603ef6d`
(26 files). Combined archive SHA-256
`825051352d971f2a4db371d272d113947f15ee36e033609e7aacb276f7446be8`
(48 files plus three canonical paired-matrix proofs). The local verifier checks
every hash, exact executed sources, trained pair identities, source snapshots,
terminal commands and monitors, full output/F64 gates, common warm durations,
all ABBA windows and independently recomputed timing means.

```text
python make_joined_global_alpha_probe.py
python3 launch_joined_global_alpha_probe.py ROOT ROOT/logs/joined-global-alpha-probe-20261014
python3 launch_joined_global_alpha_probe.py ROOT ROOT/logs/joined-global-alpha-probe-confirm-20261014
python3 archive_joined_global_alpha_probe.py ROOT ARCHIVE
python verify_joined_global_alpha_probe.py RESULTS REPORT_JSON
```
