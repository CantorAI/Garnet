# Optional full-model integration: initial full run verified

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

All original 19 trained-model gates completed at PID 68935 with exit 0 in
1203.99 seconds. The first added session suite at PID 71873 failed on an XLang
lambda/comprehension scope error (`NameError: row`), before any speed worker.
Its 908 archived files are independently verified in `failure-verification.json`.
The reference argmax now uses explicit loops with unchanged lowest-index
semantics; no compiled source, numerical bound or library was changed.

Repaired PID 90326 exited 0 in 164.92 seconds, including six trained B1/B2
session/fallback/nonfinite-recovery cases and all 72 complete matching requests
in six native ABBA profiles. Independent verification covers 1,221 files, all
19 original stages, six recovery cases, every complete normalized output and
all actual fusion counts. Each candidate prediction reports one fused dispatch;
the unchanged global-alpha control reports zero. The compiled candidate SHA-256
is `8cfcfb392d19fad68695ad7d80c8c3b5989f01ca956fa50370c7b2f960230177`.

| Model / batch | Control row tokens/s | Candidate row tokens/s | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 58.1794 | 59.3065 | 1.0194 | 0.9999 |
| Text / 2 | 57.6349 | 58.5298 | 1.0155 | 1.0140 |
| Coder / 1 | 72.5366 | 72.8213 | 1.0039 | 1.0284 |
| Coder / 2 | 70.6987 | 70.9894 | 1.0041 | 0.9997 |
| VL / 1 | 48.2638 | 48.9986 | 1.0152 | 0.9665 |
| VL / 2 | 44.3528 | 45.1830 | 1.0187 | 1.0077 |

Initial decode direction is positive but small (0.39–1.94%); TTFT is mixed.
These are native control/candidate ratios, not new vLLM ratios. Warm means use
four requests/policy, first requests excluded. Initialization and first request
AFTER initialization are separately retained in `inference-verification.json`;
caches are not flushed, and their clocks must not be summed into a claimed
hardware-cold end-to-end measurement.

Full six-profile confirmation PID 93682 is running at
`ROOT/logs/fused-head-argmax-confirm-native-20261015`. Live status must be checked
against the actual handle. Production selection remains unchanged pending
repeat measurements and assessment of the mixed TTFT behavior. Reference
libraries and the original GPT-OSS and Pi checkouts remain preserved.

Initial-failure archive SHA-256:
`d1c7c7cea9681d70832c29205f3ca8bbd69eb24a8cc3557dcf3184880b6c8712`.
Successful initial inference archive SHA-256:
`0259b835c0a8e0784ba547188c0bf636e9ecd6acf99366e7f7d0f752d4efa611`.

```text
python3 prepare_fused_head_argmax_sources.py
python3 install_fused_head_argmax_sources.py ROOT
python3 launch_fused_head_argmax_checks.py ROOT ROOT/logs/fused-head-argmax-checks-20261015 checks
python3 queue_fused_head_argmax_profiles.py ROOT ROOT/logs/fused-head-argmax-native-20261015
python3 launch_repaired_fused_head_argmax_profiles.py ROOT ROOT/logs/fused-head-argmax-native-repair-20261015
python3 archive_fused_head_argmax_inference.py ROOT ARCHIVE fused-head-argmax-native-repair-20261015
python -X utf8 verify_fused_head_argmax_inference.py EVIDENCE VERIFICATION_JSON fused-head-argmax-native-repair-20261015
python3 launch_confirm_fused_head_argmax_profiles.py ROOT ROOT/logs/fused-head-argmax-confirm-native-20261015
```

The full 2x decode / half-TTFT goal remains unmet. Archive and independently
verify the repeat before selecting the candidate or issuing any new matched
comparator claim.
