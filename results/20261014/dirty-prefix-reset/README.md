# Bounded cache reset: repeated component signal

Clearing only the previously written KV prefix substantially reduces reset
cost in a synthetic component on the actual M4. This is not yet integrated
into Garnet serving and does not prove a TTFT, decode or vLLM improvement.
Selected libraries and production source remain unchanged.

The current full-request telemetry records text batch-2 reset taking about
20–25 ms. This experiment compares temporary zero-vector allocation/copy
(current baseline), full direct CPU zero, and bounded direct CPU zero. Actual
checkpoint configurations determine all state sizes at capacity 256 and B1/B2.
Each KV head retains its full allocation; only its dirty token prefix is
cleared. All convolution and recurrent state is fully cleared in every policy.
FP32 state storage, MPS row strides and padding remain unchanged.

| Model / batch, 64 dirty tokens | Initial legacy / full-direct / prefix ms | Confirmation legacy / full-direct / prefix ms |
| --- | ---: | ---: |
| Text / 1 | 4.437 / 0.609 / 0.138 | 4.092 / 0.703 / 0.156 |
| Text / 2 | 19.633 / 1.651 / 0.368 | 19.357 / 1.432 / 0.642 |
| Coder / 1 | 0.309 / 0.052 / 0.049 | 0.310 / 0.052 / 0.050 |
| Coder / 2 | 1.108 / 0.111 / 0.112 | 1.117 / 0.156 / 0.134 |
| VL / 1 | 2.605 / 0.447 / 0.365 | 2.862 / 0.336 / 0.251 |
| VL / 2 | 8.795 / 0.745 / 0.607 | 8.955 / 0.610 / 0.447 |

These are reset-component wall means after bounded GPU writes, not full-request
estimates. Differences between component and serving cache behavior can be
large: the earlier full direct-zero experiment reduced TTFT but slowed text
batch-2 decode by 1.7–2.4%. Bounded clearing must therefore pass native repeats
before selection; component savings must not be added to old comparison results.
Coder gains over a full direct clear are small or absent.

Both runs cover 25 profiles: all three models × B1/B2 × dirty lengths
0/1/64/256, plus a padded synthetic allocation. Each run validates 300
nonzero-to-positive-zero transitions on CPU and GPU and records 3,600 timed
resets. The prefix-zero case still clears all linear state. Full-capacity
cases exercise fallback-equivalent work. Whole logical buffers, including
unwritten regions, finish at positive zero and padding retains its sentinel.

The first run checked initialization and complete post-clear buffers but did
not separately observe the GPU producer pattern. The confirmation adds that
independent observation before every numerical clear: every dirty element
matches the known finite nonzero pattern and every unwritten element is zero.
Its local audit requires `all_gpu_dirty_patterns_observed: true`. The actual
reset policies and timing scope are unchanged. No tolerance is involved.

Each profile alternates all three policies while warming for at least 250 ms,
then runs two rounds of legacy/direct/prefix/prefix/direct/legacy. Each window
has three warm operations and twelve samples. Fresh bounded GPU writes and
completion precede every reset; their cost and correctness checks are excluded
from reset timing. Initial full initialization is excluded for every policy.
OS caches and GPU frequency are uncontrolled. The independent 0.5-second
guard rejects other inference/probe jobs and only terminates owned processes.

Initial owned PID 42202 exited 0 in 37.92 seconds. Confirmation PID 43002
started after it was terminal and exited 0 in 42.46 seconds. Each archive has
16 files, exact executed sources, compiler commands, monitors and a hash/size
proof for the omitted binary. Local audits independently reconstruct model
state sizes, all profile/check/window/sample coverage and timing means.

Archive SHA-256:

- Initial: `4bf3cb9bf30bde60a8267d1d5da71682bad35befab78c1aed9c913e1ff0ca223`
- Confirmation: `076071956340e461e6657a41f1b8062003f2b84a51477190fb0d630d59ab2850`

```text
python3 launch_dirty_prefix_reset_probe.py ROOT ROOT/logs/dirty-prefix-reset-probe-20261014
python3 launch_dirty_prefix_reset_probe.py ROOT ROOT/logs/dirty-prefix-reset-confirm-20261014
python3 archive_dirty_prefix_reset_probe.py ROOT JOB ARCHIVE
python verify_dirty_prefix_reset_probe.py RESULTS VERIFICATION_JSON
```

## Native admission requirements

Any native implementation must default off, preserve original selected builds,
and track the high-water mark of validated active KV writes independently for
each session. New buffers require full zero initialization. Generic graph
outputs, unknown state writers, unsupported cache layouts, failed commands
and poisoned state require a full reset. Inactive rows must not advance dirty
extent. Gaps and overwrites still require clearing through the highest actual
write, not merely the number of generated tokens. Extent must persist across
chunk profile changes and swaps of private cache storage. Linear state remains
fully reset. Full-cache snapshots after reset must match the original reset
bit for bit, with no reuse of prior request contents.

Validation must cover all original nineteen numerical stages, multiple
request lengths, sparse/overwritten/inactive slots, two interleaved private
sessions, failed-run recovery and full-capacity boundaries before six-profile
native ABBA timing. A repeated serving gain and fresh matched vLLM comparison
are required before any selection change. The full 2x/half-latency goal is unmet.
