# Reset and dependent GPU read in one command buffer: repeated component signal

Two actual exclusive M4 runs exited 0 in 9.23 and 9.16 seconds. Independent
verification covers 35 archived files, 126 CPU/GPU positive-zero and padding
checks, and 1,344 timed executions. All six actual-config model/batch profiles
improve over the legacy zero-vector allocation/copy plus dependent GPU read.
This is not a measured TTFT or full-model gain.

The previous GPU-clear experiment submitted and waited for a reset command
separately. Here the candidate puts row-preserving blit fills and a dependent
GPU read in the same command buffer. The control performs the current CPU
allocation/copy reset before the same read. Both observed wall clocks include
reset work, read encoding, submission and completion. GPU conditioning writes
and waits happen before each sample and are excluded, as is CPU validation.
The GPU read checks positive-zero bits across every logical state element;
it is a synthetic dependent read, not a trained inference operation.

| Model / batch | Initial legacy → joint wall ms | Repeat legacy → joint wall ms | Repeat joint / legacy |
| --- | ---: | ---: | ---: |
| Text / 1 | 5.2963 → 1.6588 | 5.0763 → 2.1955 | 0.4325 |
| Text / 2 | 23.6289 → 3.7592 | 22.8792 → 4.0736 | 0.1780 |
| Coder / 1 | 1.0304 → 0.7649 | 0.7319 → 0.5124 | 0.7001 |
| Coder / 2 | 2.3989 → 1.3672 | 1.7197 → 0.7415 | 0.4312 |
| VL / 1 | 4.3236 → 1.3616 | 3.7643 → 0.9833 | 0.2612 |
| VL / 2 | 10.4411 → 2.4609 | 10.4529 → 2.7948 | 0.2674 |

Actual checkpoint configuration hashes determine all FP32 state shapes at
capacity 256 and B1/B2, including convolution/recurrent state. A padded 3-row
synthetic case preserves every padding byte but has mixed performance
(0.9291 / 1.1018 wall fractions); this is not a universal small-buffer win.
Direct CPU memset is recorded as a third diagnostic policy and can beat the
joint candidate in several profiles. GPU spans increase with the added fill
work; reduced total wall time does not imply a faster GPU read kernel.

Each profile uses at least 250 ms paired warm-up, then two balanced
legacy/direct/joint/joint/direct/legacy rounds, eight samples per window.
The source, raw configs/profiles, commands, exclusive guards, all windows,
omitted binary hashes and independent calculations are in `evidence/`.
Archive SHA-256:
`8573262d9b96aa9e8e821cc2f4ab3c709ccaeb9aef6c463a021f2b81e6c808f9`.

```text
python3 launch_joint_state_zero_probe.py ROOT ROOT/logs/joint-state-zero-20261015
python3 launch_joint_state_zero_probe.py ROOT ROOT/logs/joint-state-zero-confirm-20261015
python3 archive_joint_state_zero_probe.py ROOT ARCHIVE
python -X utf8 verify_joint_state_zero_probe.py EVIDENCE VERIFICATION_JSON
```

Next required work is an optional deferred-reset implementation. Reset must
still expose logical positive-zero snapshots immediately; physical clearing
must finish before any GPU consumer in the next command buffer. Input/output
ownership, host updates, canceled resets, aborts, partial submissions and
poisoned-session recovery need validation before all original model gates and
matched repeated inference trials. No production reset semantics are changed
by this component experiment. The complete 2x decode / half-TTFT goal remains
unmet; protected GPT-OSS and Pi checkouts remain unchanged.
