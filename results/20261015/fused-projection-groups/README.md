# Single-dispatch independent projections

This component experiment preserves the native BF16 projection dot products and
places independent projections in disjoint ranges of one GPU dispatch. It is
distinct from the earlier serial/concurrent compute-pass study, which retained
multiple dispatches, and from gate/up/SiLU fusion, which changed the operation
boundary. There is no serving integration or speedup claim yet.

The generator retains the original two-/four-output dot-product bodies and
passes the original parameters, input, weights and separate outputs to them.
One grid covers the sum of the projection grids. A group-uniform branch selects
the projection and adjusts its group index. The combined dispatch uses 128
threads; small projections retain only their original two active SIMD groups.
No learned weight is repacked, no sum is split, and precision is unchanged.

The experiment covers actual layer-zero learned MLP gate/up pairs for all three
models, separate dense Q/K/V projections for text and coder, and irregular
synthetic read-only-input pairs, at B1/B2. The harness requires complete output
and padding bits to match the original kernels for fresh original, negated,
halved and doubled inputs, including negative-zero values. The original FP64
gamma(ceil(K/32)+32)*sumabs+1e-7 envelope remains unchanged. Timing uses common
250 ms warmup and two ABBA rounds, twelve samples per window. Compile and kernel
pipeline creation are outside component execution timing.

Owned queue PID 47884 waits for exact owned inference PID 43673 at
`logs/threaded-striped-inference-20261015` to terminate successfully before it
compiles or uses the GPU. Its destination is
`logs/fused-projection-groups-20261015`. The same exclusive-process, disk-space
and 0.5 s monitor guards apply. Neither a queue record nor generated code proves
successful execution; actual terminal evidence and independent audit are required.

The unchanged original encoder path is the control. Component timing includes
encode, submission and wait, so it cannot establish a model TPS or TTFT gain.
Repeated component benefit would only justify considering a separate full-model
integration and output-validation study. The full goal remains unmet and active.

## Actual initial result: mixed; not integrated

PID 47884 terminated exit 0 after 243.8145 seconds, including its dependency
wait. The independent audit verified 30 archived files, 16 group/batch checks,
12,738 complete-output validated executions and 1,536 timed executions. Every
candidate output and padding bit matches; the worst original FP64 envelope
ratio is 0.0090262. Compilation and all three model stages passed their exclusive
guards. Thus the dispatch is compatible, but compatibility is not a speed win.

| Model | Batch | Group | Component wall fraction | GPU fraction |
| --- | ---: | --- | ---: | ---: |
| Text | 1 | gate/up | 1.01 | 0.99 |
| Text | 2 | gate/up | 0.98 | 0.99 |
| Text | 1 | Q/K/V | 1.01 | 0.96 |
| Text | 2 | Q/K/V | 0.90 | 0.91 |
| Coder | 1 | gate/up | 0.96 | 0.91 |
| Coder | 2 | gate/up | 1.06 | 1.07 |
| Coder | 1 | Q/K/V | 1.05 | 1.03 |
| Coder | 2 | Q/K/V | 0.97 | 0.91 |
| VL | 1 | gate/up | 1.09 | 1.05 |
| VL | 2 | gate/up | 1.03 | 1.00 |

Fractions below one are faster. These rounded component values are not whole
model TPS/TTFT. Text B2 Q/K/V improves, but VL gate/up and several coder groups
regress. The initial mixed result does not justify a blanket serving change.
There is no second trial or full inference integration for this candidate.
Irregular-pair results and full precision values remain in `verification.json`.

Archive SHA-256:
`4bcce4ed75a37c04ab4f7c9b7f38579a44a4bec65f8121a62ca45fdd3d600da5`.

```text
python3 queue_fused_projection_groups.py ROOT ROOT/logs/fused-projection-groups-20261015
python3 archive_fused_projection_groups.py ROOT ARCHIVE
python -X utf8 verify_fused_projection_groups.py RESULTS VERIFICATION_JSON
```
