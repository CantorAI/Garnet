# One joined product for lossless input components: rejected

The M4 component preserves every supported FP32 input bit through three
scaled FP16 components, joins them as 3×B rows and encodes one MPS matrix
product. FP32 partial results are recombined and rescaled. All eligible
numerical checks pass, but the candidate is slower across all six unweighted
large-matrix pools. It is rejected for serving integration; no full inference
or vLLM gain is claimed. No serving source or selected library changes.

The earlier three-product prefill method was substantially slower. This new
trial tests B1/B2 decode with one joined product against the exact original
native BF16 shader, including its selected small-output policy. The control
uses its original default compiler options; decomposition/recombination
disable fast math. Learned BF16 values restore bitwise from scaled-half
storage, and GPU reconstruction checks every FP32 input bit, including signed
zero. Arithmetic order changes. FP32 result buffers alone are not proof of
an internal accumulation implementation; independent FP64 and unchanged
control bounds gate all outputs. No approximation or wider tolerance is used.

The successful run validates 64 matrix/batch profiles from the three frozen
checkpoints, including trained first-layer products, complete eligible heads
and irregular synthetic cases. It performs **51,750 validated executions**
and **6,144 timed executions**. Four fresh power-of-two input variants and
output padding are checked. Worst candidate FP64 normalized error is
0.0075366 under the original `gamma(ceil(K/32)+32)*sumabs+1e-7` envelope.
Worst control-comparison ratio is 0.0686387 under `5e-6+5e-5*abs(control)`.
Wide-exponent input rejection passes separately for every model.

| Unweighted trained large-matrix pool | Batch | Candidate/control GPU fraction | Wall fraction |
| --- | ---: | ---: | ---: |
| Text | 1 | 1.1685 | 1.1316 |
| Text | 2 | 1.2031 | 1.1466 |
| Coder | 1 | 1.1032 | 1.0975 |
| Coder | 2 | 1.1650 | 1.1473 |
| VL | 1 | 1.1075 | 1.1189 |
| VL | 2 | 1.1038 | 1.0896 |

Fractions above one are slower. Pools sum measured times once per collected
trained matrix with N>256; they are not full-model layer-weighted estimates
or tokens/sec. A small coder up-projection GPU signal has essentially flat
wall time and does not justify a favorable-number repeat. No DRAM traffic or
internal MPS kernel count is inferred merely from using one API operation.

Two weight matrices fail exact-half admission: Qwen3's vocabulary embedding
head (`matrix-0`) and Qwen3.5 layer-0 QKV (`matrix-3`). They are explicitly
excluded and would require the original native fallback. No row is rounded,
clamped or silently forced into the candidate. Every eligible result is
checked, with all weights/matrix identities pinned to the original checkpoint.
These are deterministic synthetic activations, not model hidden-state evidence.

Every eligible profile alternates common warming for at least 250 ms, then
two ABBA rounds with three warm operations and twelve measured samples per
window. Timers include GPU decomposition, MPS product and recombination, plus
separate host encoding/wall intervals. Setup, uploads and numerical readbacks
are excluded equally. Native reset initial/confirmation jobs were both
terminal before this trial; guards reject other inference/probe jobs every
0.5 seconds and only stop owned processes. Caches/frequency are uncontrolled.

Initial PID 95000 exits 1 in 5.62 seconds before real timings. Its inherited
unsafe-input fixture used index 101, absent from the first eligible short
matrix. The repaired rejection-only branch injects 2^-60 at index 1. Timed
inputs, numeric kernels and tolerances remain unchanged; the independent audit
verifies the one-line source difference. Repaired PID 95450 exits 0 in
46.69 seconds, including all three rejection gates and all eligible profiles.
Both terminal runs, exact executed sources, commands, monitors and omitted
binary hashes are archived. No favorable performance sample was selected.

Combined archive SHA-256:
`62aef0fd036552ccd5aed0f341316b2876a35062545be7958f3ed421e02b6d42`.
The initial failed archive was
`3a5a89fe6b39ffb91e2769f23935ca4cff0b883066e69d4588f1624ad85c1fbc`.
`verification.json` verifies all 47 files, source repair, exclusions, coverage,
numeric gates and independently recomputed raw ABBA means.

```text
python3 launch_joined_activation_decode_probe.py ROOT ROOT/logs/joined-activation-decode-probe-20261014
python3 launch_joined_activation_decode_probe.py ROOT ROOT/logs/joined-activation-decode-repair-probe-20261014
python3 archive_joined_activation_decode_probe.py ROOT ARCHIVE
python verify_joined_activation_decode_probe.py RESULTS VERIFICATION_JSON
```

The full three-model 2x decode / half-TTFT goal remains active and unmet.
