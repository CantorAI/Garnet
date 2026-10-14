# Decode-only fused head: experiment in progress

The all-head variant repeated a 0.39–2.26% whole-model decode improvement,
but text B2 first-token latency regressed in both runs. This experiment keeps
the existing projection/argmax path when the session's physical chunk is
larger than one, and fuses only single-token runs. Agreed serving prefill
profiles use chunks of at least two; a direct single-token prefill API call
would use fusion as well. No precision, checkpoint, resolution, prompt, output
cap, request cache or prediction semantics are changed.

`GARNET_METAL_FUSED_HEAD_DECODE_ONLY` is OFF by default and requires the existing
optional fused-head flag. The previous candidate library and archived results
are preserved. Five route/measurement files are installed only after verifying
their exact original hashes. Raw fused shader arithmetic is unchanged.

Fresh build `build-decode-only-head` has passed both actual CTests, including
the full-bit/FP64 fused-kernel, layout, ownership and recovery checks. All
original 19 model stages are being rerun under owned PID 96768 at
`ROOT/logs/decode-only-head-checks-20261015`. PID 98218 waits for successful
terminal validation before all six trained session cases and the six matched
native ABBA profiles. Recheck actual handles for live state.

The benchmark asserts zero fused dispatches for prefill's first prediction
and exactly one for each subsequent decode prediction. Every complete output,
prompt ID and finished flag must match; images must be freshly encoded. All
three models and both batches remain in scope. No end-to-end improvement is yet
established for this routing policy, and no new vLLM comparison is claimed.

```text
python3 prepare_decode_only_head_sources.py
python3 install_decode_only_head_sources.py ROOT
python3 launch_decode_only_head_checks.py ROOT ROOT/logs/decode-only-head-checks-20261015 checks
python3 queue_decode_only_head_profiles.py ROOT ROOT/logs/decode-only-head-native-20261015
python3 archive_decode_only_head_inference.py ROOT ARCHIVE
python -X utf8 verify_decode_only_head_inference.py EVIDENCE VERIFICATION_JSON
```

The full 2x decode / half-TTFT goal remains active and unmet. The GPT-OSS and Pi
checkouts remain unchanged. The next decision requires actual complete-output
measurements and repeated warm/cold observations, not the earlier component
gain or a different comparator baseline.
