# Full-shape BF16 decode specialization: head regressions, mixed body gains

Two actual sequential component trials exited 0. The independent audit verified
50 archived files, 34 canonical matrix identities, 136 matrix/batch configurations,
142,836 bit-exact validated executions and 26,112 timed executions. Arithmetic,
learned values and precision are unchanged. No serving change is selected and
no new inference or vLLM speedup is claimed. The full goal remains unmet.

Unlike the earlier single-row experiment, this specializes batch, input width,
output width and both physical strides through Metal function constants. Mode 1
retains all bounds; mode 2 additionally removes output-tail branches only when
N is divisible by the selected kernel's complete tile width, 4 for small-output
and 16 otherwise. Original shaders and default compilation remain the control.
All 34 learned/synthetic fixtures run at B1/B2 with fresh original, negated,
half and doubled FP32 inputs. Every warm/timed result and padding bit matches
the original. The independent FP64 maximum error/bound ratio is 0.0100017 under
the original gamma(ceil(K/32)+32)*sumabs+1e-7 bound.

The native comparison measures mode 2 versus the original; the dimensions
comparison measures mode 2 versus dimension-only mode 1. Both retain two ABBA
rounds, 12 samples per window and at least 250 ms common checked warmup. Raw
arrays and both comparisons are independently recomputed in `verification.json`.

The following are unweighted trained-matrix component wall-time pools, with
synthetic fixtures excluded. The vocabulary head is separate from the first
decoder-layer body matrices. Fractions above 1 are slower. These are not model
TPS or TTFT ratios, and the single sampled layer is not a whole-model weighting.

| Model | Batch | Body candidate/control, initial / repeat | Head candidate/control, initial / repeat |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 1.02904 / 0.98322 | 1.05425 / 1.05790 |
| Qwen3-0.6B | 2 | 1.07293 / 0.95672 | 1.04704 / 1.04993 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.97121 / 1.00500 | 1.11790 / 1.13078 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.96546 / 0.96922 | 1.03133 / 1.01921 |
| Qwen3.5-0.8B | 1 | 0.96907 / 0.94224 | 1.01769 / 0.98809 |
| Qwen3.5-0.8B | 2 | 0.99904 / 0.97018 | 1.03958 / 1.04548 |

A blanket all-projection change is not justified: text body changes direction,
coder B1 body changes direction, and all text/coder heads slow in both trials.
Coder B2 body and VL B1 body have repeated lower times. A body-only serving
candidate could test whether these effects survive a real graph, but that has
not been implemented or measured by this study. No cache, occupancy or register
cause for these timing changes has been proven.

Pipeline creation is outside component projection clocks. Creating two
specialized pipelines for each of 68 matrix/batch configurations took an
observed summed 1048.5767 ms initially and 19.0287 ms on repeat, maximum single
configuration 69.9800/0.5993 ms. Compiler/OS caches are not flushed; this is not
a hardware-cold measurement. Any serving experiment must include this setup
in initialization or first-use costs rather than hiding it.

Actual owned jobs:

| Trial | PID | Exit | Wall seconds |
| --- | ---: | ---: | ---: |
| shape-specialized-bf16-20261015 | 86193 | 0 | 42.2538 |
| shape-specialized-bf16-confirm-20261015 | 86986 | 0 | 40.1661 |

The initial trial was independently audited before the repeat. No comparator
or other owned GPU probe runs in parallel. The same 0.5-second guards reject
unknown inference/probe activity. GPT-OSS and Pi checkouts and the selected
serving library remain unchanged.

Initial archive SHA-256:
`49bba312f34ea1b9cec09add31e7a90fab79b1d542891e5c29e29b4c5f409aa7`.
Combined archive SHA-256:
`2a4814da1a5ad5b41fb4a4d4ac1c417105be313bd85be8f5e698ea09e0095ddb`.
`evidence/` retains raw logs, source, compiled shader text, commands, versions,
process monitors and manifests. Checkpoint arrays and binary are omitted with
hashes retained. `pools.json` separates both comparisons and setup costs.

```text
python3 launch_shape_specialized_bf16.py ROOT ROOT/logs/shape-specialized-bf16-20261015
python3 archive_shape_specialized_bf16.py ROOT ARCHIVE
python -X utf8 verify_shape_specialized_bf16.py EVIDENCE VERIFICATION_JSON
python -X utf8 summarize_shape_specialized_bf16.py VERIFICATION_JSON POOLS_JSON
```
