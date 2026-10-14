# Retained CPU zero source: repeated component signal

Two actual exclusive M4 runs exited 0 in 9.28 and 9.19 seconds. Independent
verification covers 41 archived files, 126 positive-zero/padding checks and
1,344 timed executions. A retained zero vector reduces reset-plus-dependent-
GPU-read wall time against the current temporary-allocation/copy path in all
six actual-config model/batch profiles. No full-model latency or decode gain
has been measured for this policy.

Only the CPU reset source changes: one FP32 zero vector sized for the largest
state is allocated per profile before timing and reused by the existing row-
preserving memcpy loop. The GPU reader and all numeric/padding gates remain
unchanged. The initial retained-source allocation is excluded from component
timing; a serving implementation must account for it in initialization.

| Model / batch | Initial legacy → retained source ms | Repeat legacy → retained source ms | Repeat fraction |
| --- | ---: | ---: | ---: |
| Text / 1 | 5.4990 → 2.8531 | 4.9615 → 3.3401 | 0.6732 |
| Text / 2 | 24.8753 → 6.3341 | 22.1480 → 6.8148 | 0.3077 |
| Coder / 1 | 0.7284 → 0.5552 | 0.8694 → 0.6924 | 0.7965 |
| Coder / 2 | 2.3387 → 1.3562 | 2.1763 → 1.1828 | 0.5435 |
| VL / 1 | 4.2853 → 1.8704 | 3.7536 → 1.6714 | 0.4453 |
| VL / 2 | 9.9708 → 3.6501 | 9.4907 → 3.6834 | 0.3881 |

Both runs reduce wall time by 20.4–74.5% over model-sized profiles. A tiny
padded case is mixed (1.0792 / 0.9033 fractions), so no universal small-buffer
claim is made. Same-command GPU clearing remains the third diagnostic policy;
its results do not establish inference behavior for this CPU source policy.

Actual checkpoint configs determine all FP32 state sizes at capacity 256 and
B1/B2. GPU conditioning writes/waits occur before each sample and are excluded.
Wall timing includes reset work, dependent GPU-read encoding, submission and
completion; CPU validation is outside. Every logical element is checked for
positive-zero bits on CPU/GPU, and padding retains its sentinel. Each profile
has a 250 ms minimum warm-up, then two balanced three-policy forward/reverse
rounds of eight samples/window. Raw commands, configs, source generation,
compiled-binary omissions/hashes, guards and all windows are in `evidence/`.

Archive SHA-256:
`df2f292b28660d660f38496dd1f170b63914e450389db992ef2d01eca4dcb8a7`.

```text
python3 launch_reused_zero_source_probe.py ROOT ROOT/logs/reused-zero-source-20261015
python3 launch_reused_zero_source_probe.py ROOT ROOT/logs/reused-zero-source-confirm-20261015
python3 archive_reused_zero_source_probe.py ROOT ARCHIVE
python -X utf8 verify_reused_zero_source_probe.py EVIDENCE VERIFICATION_JSON
```

Next required work is an optional per-session retained zero source, preserving
the original CPU copy and reset/snapshot semantics. Validate initialization,
rebuild/close and private-session ownership, then all original model gates and
complete-output warm/cold inference repeats. The hoped-for absence of the
earlier decode penalty is unproven. Selected libraries, protected checkouts
and the complete 2x / half-latency goal remain unchanged; that goal is unmet.
