# Optional full-model integration: validation in progress

The component gain repeated, so `GARNET_METAL_FUSED_HEAD_ARGMAX` now enables
an optional greedy-only native BF16 path. The option is OFF by default. Generic
compiled-graph execution still returns complete logits. Unsupported batches,
float heads and unsupported pipeline capabilities retain the original path.
Every vocabulary dot product is evaluated, with FP32 accumulation; no pruning,
quantization, prefix reuse or speculative predictions are introduced.

The fused path honors physical MPS row strides for both partial winners and
debug logits, retains scratch and output ownership through completion/abort,
detects nonfinite logits, and preserves lowest-index ties. Full-logit debug
emission is available to the numeric CTest and is disabled in generation.
Context/session counters record actual use, reset per run/reset/abort, and
are included for every prefill and decode prediction in the benchmark records.
If GPU execution completes but CPU validation rejects nonfinite predictions,
the failed session retains its observed dispatch count until reset; it remains
poisoned and cannot serve another request without reset.

Installed source is pinned in `fused-head-argmax-staging/manifest.json` on the
investigation Mac; ten original files were hash-checked and preserved before
twelve candidate files were installed. The first installer call failed because
its reusable installation helper was absent. Uploading that unchanged helper
resolved this before any candidate source was installed.

Actual CTest observations so far: `metal_fused_head_argmax_reference` and
`metal_mps_matmul_reference` both passed (3.53 seconds total). The new CTest
covers 56 full-bit/independent-FP64 numeric cases, three full vocabulary shapes,
irregular tails/padded strides, B1/B2, production emission-off predictions,
lowest-index ties, signed zeros, nonfinite inputs, unsupported batch fallback,
uninitialized/invalid arguments, pending-output ownership and abort recovery.

The full original 19 trained-model gates are still required. Owned validation
PID 68935 runs at `ROOT/logs/fused-head-argmax-checks-20261015`; queued PID 71873
waits for its successful terminal result before three trained session/fallback/
poison recovery suites, then six-profile complete-output native ABBA trials.
Live status must be rechecked against the actual processes, not this document.
No new whole-model speed or vLLM ratio is established yet. Reference libraries
and the original GPT-OSS and Pi checkouts remain preserved.

```text
python3 prepare_fused_head_argmax_sources.py
python3 install_fused_head_argmax_sources.py ROOT
python3 launch_fused_head_argmax_checks.py ROOT ROOT/logs/fused-head-argmax-checks-20261015 checks
python3 queue_fused_head_argmax_profiles.py ROOT ROOT/logs/fused-head-argmax-native-20261015
```

The full 2x decode / half-TTFT goal remains unmet. After all jobs end, archive
and independently verify every original gate, trained recovery case, complete
output, count and warm/cold observation. Repeat any measured whole-model gain
before selecting the candidate or issuing a new matched comparator claim.
