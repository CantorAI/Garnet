# QV output views without a scatter dispatch

Final component status: reject under the original complete endpoint rule.
Canonical shared output views remove the Q/V scatter encode, and all numerical
and buffer-guard checks pass, but no model meets the required GPU and wall
improvement at every endpoint. No unchanged repeat or native integration is
admitted.

The control is the accepted d52 context with independent original Q/K/V
projections. Keep the original K function, previous triple admission, scale
selection and exact Q/V weight packing. Allocate one joined output with an
extra guard row, then create canonical Q and V MPSMatrix views of its disjoint
column ranges. Encode one joined Q/V product directly into that buffer and
preserve the independent K output. No candidate scatter is encoded.

Download is adapted to honor the MPSMatrix byte offset. Other GPU consumers,
host writes and graph/session ownership are not adapted by this component
prototype. Its output values and timing therefore do not establish a usable
native model path. The previously compiled scatter pipeline is retained but
unused; the generated candidate encode body is audited to contain no scatter
encoder call. A future native implementation would require complete consumer,
public-output, alias and lifetime validation as well as original model gates.

## Actual execution and numerical evidence

Primitive PID 44008 exits 0 in 11.285164042 s. The 24 changed dyadic cases cover
all six row counts, unequal Q/V widths, canonical offsets, original key bits,
guard rows, pending read rejection, foreign writer rejection, abort/recovery,
fallbacks and bounded 64-group/2-reject ownership. The synthetic duplicate Q/V
weight buffer remains 256 bytes. [Primitive audit](primitive/verification.json).

Learned PID 44331 exits 0 in 83.593469917 s. All 116 profiles, 464 changed
structured-input cases, 52 original FP32-accumulation key matrices and 16
fallback cases pass. Numerical checks cover 177,553,408 reference values,
48,758,784 exact output-view bits and 10,723,328 original-K output bits. Every
candidate validation checks the physical guard row. Candidate/control error
ratio is zero on these structured inputs; arbitrary activations and native
consumers remain outside this proof. [Learned audit](initial/verification.json).

## Complete fixed timing result

Keep the previous first/last eligible layers, both batches, 2 s alternating
warmup, own-null followed by comparison, 10 ABBA rounds, 3 warm calls and 5
measured samples per window. Retain all 4,800 samples. Require >=5% GPU AND
wall gain at all four endpoints of a model, with each own-null within 5%.
No model passes. Five individual profiles pass, which cannot rescue the
complete model rule. [Decision](initial/decision.json).

All fractions are candidate/control. Own-nulls exclude unstable profiles;
they never normalize the measured comparison.

| Model | Layer | Rows | GPU fraction | Wall fraction | Null GPU fraction | Null wall fraction | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 26 | 0.819664 | 0.875334 | 0.952410 | 0.965325 | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 52 | 0.969518 | 0.938678 | 0.998734 | 0.992928 | No |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 26 | 0.813070 | 0.892107 | 0.964992 | 0.998014 | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 52 | 0.902872 | 0.900946 | 1.043498 | 1.043898 | Yes |
| Qwen3-0.6B | 0 | 24 | 0.948205 | 0.947382 | 1.033279 | 1.028587 | Yes |
| Qwen3-0.6B | 0 | 44 | 0.903088 | 0.881389 | 1.017856 | 1.018869 | Yes |
| Qwen3-0.6B | 27 | 24 | 0.973003 | 0.976237 | 0.990495 | 1.004954 | No |
| Qwen3-0.6B | 27 | 44 | 0.958871 | 0.956137 | 0.955636 | 0.948814 | No |
| Qwen3.5-0.8B | 3 | 80 | 0.974693 | 0.950118 | 1.005895 | 1.022666 | No |
| Qwen3.5-0.8B | 3 | 160 | 0.995519 | 0.978377 | 0.987184 | 0.997336 | No |
| Qwen3.5-0.8B | 19 | 80 | 0.960265 | 0.938424 | 0.985188 | 0.985614 | No |
| Qwen3.5-0.8B | 19 | 160 | 0.969132 | 0.950568 | 0.966780 | 0.975907 | No |

The text first-layer profiles pass, while last-layer profiles miss the gain
rule and the last-layer batch-2 null wall fraction is below 0.95. Coder misses
first-layer batch-2 GPU gain. VL GPU improvements remain below 5%. No filtering,
null normalization, threshold relaxation or favorable repeat is used.

Duplicate weight estimates remain text 168 MiB, coder 42 MiB and admitted VL
36 MiB, with additional output guard-row storage per prepared shape. All setup
clocks and peak RSS are component-only. [Captured costs](cost-summary.json).
The separately verified integer inverse is not used.

Frozen archive SHA256 values:

- Primitive: `6df7a4e0c84e78725e29bcc6f7773fbd0166b55033cc3ea868acd4bd61350a1c`.
- Complete learned cohort: `131b7c373fbd7f8ca77a46b101ad1a9ee8aecc692ab8b4d48446a92c6cc84026`.

Both archives are independently extracted, regenerated and audited on Mac and
Windows. No serving source, native library or matched vLLM result changes.
