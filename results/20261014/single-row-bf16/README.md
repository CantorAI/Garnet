# Single-row BF16 specialization: coder-head component signal only

The existing decode kernels conditionally update second-row accumulators but
reduce them unconditionally. This component duplicates both original kernels
with `p.rows==2` replaced by compile-time false. The original first-row float
operations, BF16 bytes, launch widths and weight address space remain unchanged.
The candidate is used only for one row; batch two dispatches the original
kernel. No production/native source route is changed.

Initial PID 2096 and fresh-process confirmation 3219 exited 0 after
36.13 / 35.79 seconds. Both validate 34 hash-verified matrix fixtures (25 trained
and nine synthetic tails), batch 1/2, four fresh input variants and padding.
All **124,786** validated executions across the two runs retain complete output
bits. Each run has 6,528 timed executions and independent FP64 checks; worst
normalized error is 0.0100017036 under the unchanged prior envelope. Each
profile uses alternating common warming of at least 250 ms, followed by two
ABBA rounds. Input uploads/checks/setup are outside the clocks.

| Model B1 | Pooled candidate/control GPU fraction, initial / repeat |
| --- | ---: |
| Text | 1.00163 / 1.00474 |
| Coder | 0.96237 / 0.98075 |
| VL | 1.00630 / 0.97181 |

Body projections are mixed and small timings fluctuate. Batch-two timing
differences occur despite identical dispatches, demonstrating that timing
variation must not be interpreted as a batch-two optimization. There is no
broad speed win or full-model inference claim.

The coder vocabulary projection (K/N 896/151936) has wall fractions
0.97910/0.95047 initially and 0.97794/0.97407 in the fresh confirmation.
GPU fractions are 0.97137/0.94913 and 0.96345/0.97706. The head direction
repeats across all four rounds. Text head is slower initially and mixed on
repeat; VL head is mixed. A future coder-head-only native experiment could
test the repeated component signal, but no integration or runtime promotion
has been performed. The overall 2x decode / half-TTFT goal remains unmet.

OS caches and GPU frequency are uncontrolled. The 0.5-second guard cannot
categorically exclude shorter external activity. Current selected runtime
libraries and protected checkouts remain unchanged.

```text
python3 launch_single_row_bf16_probe.py ROOT ROOT/logs/single-row-bf16-probe-20261014
python3 launch_single_row_bf16_probe.py ROOT ROOT/logs/single-row-bf16-confirm-20261014
python3 archive_single_row_bf16_probe.py ROOT ARCHIVE
python summarize_single_row_bf16_probe.py EVIDENCE REPORT [JOB_NAME]
```

Initial archive SHA-256:
`965bd7c16bbb2660902f18ba9632c5042434865fd40ecd2a2059035dad4cd8a7`
(26 files). Combined archive:
`cebba8f773a359a383fbeab97468d49beb55c7fdaf199958183866aea2110939`
(52 files plus 34 canonical matrix omission proofs). Verification checks every
file hash, source/shader/binary identity, learned-matrix identities, terminal
stages, all bit/padding/FP64 gates, common warm durations and independently
recomputed timing ratios. Raw evidence retains its exact bytes.
