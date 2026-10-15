# Screened head plus ordered prefill and partial cache reset

All 26 validation stages passed, followed by two six-profile native ABBA
trials. Independent audits cover 1750 archived files, the original 19 stages,
51 actual trained-head queries, six trained-head session cases, six bounded
reset/recovery cases and 144 complete matching generated outputs across the
two trials. Every actual prefill submission and reset byte count is checked.
The full 2x vLLM / half-TTFT goal remains unmet. This is a native comparison;
no fresh vLLM ratio is claimed for this combination.

The control is the validated screened-head library. The candidate additionally
enables `GARNET_METAL_PREFILL_STREAM` and `GARNET_METAL_DIRTY_PREFIX_RESET`.
Streaming submits ordered prefill work every four layers while CPU encoding
continues. Reset clears the known written KV prefix and fully clears unknown
writers and linear-state layouts. It does not reuse a prior request's prefix,
answer, or image encoding. Every new request still clears all affected state.
All three options remain OFF by default; the original selected builds and
GPT-OSS/Pi/original Mac checkouts are preserved.

## Actual warm measurements

Four warm requests per policy/profile from two fresh sequential workers in
ABBA order. Each worker's first request is excluded. TPS is per batch row;
B2 aggregate throughput is twice this rate. Prompt/image bytes, checkpoint
values, FP32 arithmetic/activations/KV, output caps and stop semantics remain
unchanged. Both VL policies use the same combined CPU frontend and freshly
encode each image. No competing inference runtime intentionally overlaps;
guards sample every 0.5 seconds.

| Model / batch | Initial control / candidate TPS | Initial control / candidate TTFT ms | Repeat candidate TPS | Repeat candidate TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 64.5593 / 63.7040 | 44.2167 / 35.2482 | 63.8128 | 35.0971 |
| Text / 2 | 62.4397 / 59.9552 | 79.9181 / 57.8403 | 60.4664 | 57.5740 |
| Coder / 1 | 78.1875 / 77.4447 | 31.6382 / 28.2757 | 78.5187 | 27.2632 |
| Coder / 2 | 76.4880 / 75.9973 | 48.5013 / 43.7541 | 75.7111 | 44.0425 |
| VL / 1 | 52.9292 / 52.9937 | 183.5340 / 176.6148 | 53.0034 | 178.2548 |
| VL / 2 | 48.3378 / 48.2917 | 345.5800 / 333.6538 | 48.3292 | 335.5571 |

| Model / batch | TTFT change initial / repeat | Decode change initial / repeat |
| --- | ---: | ---: |
| Text / 1 | -20.28% / -23.43% | -1.32% / -2.27% |
| Text / 2 | -27.63% / -26.31% | -3.98% / -2.97% |
| Coder / 1 | -10.63% / -13.92% | -0.95% / -0.46% |
| Coder / 2 | -9.79% / -10.22% | -0.64% / -0.67% |
| VL / 1 | -3.77% / -2.03% | +0.12% / -0.17% |
| VL / 2 | -3.45% / -2.27% | -0.10% / -0.07% |

All latency improvements repeat. Text gives up decode throughput; coder/VL
decode changes are smaller. This tradeoff merits further latency work rather
than declaring an all-metric win or rejecting latency gains solely because of
a small decode loss. An older vLLM cohort cannot establish a new matched ratio.

## Numerical, lifecycle and performance scope

The added native context gate checks retained FP32 bits, every rejected head
row and predictions against actual trained references. CTests cover screened
head bounds/nonfinite fallback, ties outside the seed, abort/ownership and
padding, ordered prefill submissions and bounded reset behavior. The original
trained layer, private-KV, serving, vision, exact-tail and extended-prefill
gates remain required. Both recovery suites cover B1/B2, unknown writers,
nonfinite poisoning, full reset, private state and screened close/rebuild.

`phases.json` and `confirmation-phases.json` preserve observed init, first
request after init, warm reset/prefill/decode clocks, individual child RSS and
actual counters. GPU/encode/wait clocks overlap and must not be added. Vision
includes reset. Individual child RSS is not aggregate or unique GPU memory.
OS/driver caches were not flushed, and no sum is presented as an observed
hardware-cold latency.

In the initial text B2 trial, mean decode GPU time rose from 14.6558 to
15.2886 ms while host preparation rose only from 0.1590 to 0.1682 ms.
The repeat shows the same direction, 14.6537 to 15.2121 ms GPU and 0.1592 to
0.1661 ms preparation. This locates most observed extra time in the GPU
interval; it does not establish its kernel/cache/power cause. Small host-only
tracking changes cannot recover the whole measured decode difference.

## Evidence and next work

| Job | PID | Exit | Wall seconds |
| --- | ---: | ---: | ---: |
| Original gates, trained heads and both recovery suites | 29706 | 0 | 1293.0611 |
| Dependency wait and initial full native trial | 32026 | 0 | 1342.0548 |
| Fresh-worker six-profile confirmation | 56025 | 0 | 160.4795 |

Candidate dylib SHA-256:
`75b58100ea5db4f3f1a4ee4ef18a1d3c4c0d02288a779a6bacf38ce5f91dad0e`.
Screened control SHA-256:
`2b44d93ab1e1980845be89ea59cbaedbebcb66ad385ba24df3d52af2b6bacc23`.
Initial archive SHA-256:
`2c4f37df62f3f5e4ba2419286c8c4c56e99ec3cd1ed7d84dad6e312c71630b5e`.
Combined archive SHA-256:
`e1c3fa1f033d68f036ff46815b05728bcc1d12d28f29bc0fb50fe9d711dd9ee9`.

The next investigation should address B1 body projection GPU time and the
remaining GPU prefill/vision work. Earlier FP32 MPS GEMV and broad lossless
packing experiments are already rejected and should not be repeated unchanged.
A distinct candidate is padded-row MPS body projection using the existing
lossless scaled-half storage with FP32 inputs/output, compared through
Garnet's native context. It still requires real execution, unchanged numerical
bounds and full-model validation before any speed claim or integration.

```text
python3 launch_screened_prefill_reset_checks.py ROOT ROOT/logs/screened-prefill-reset-checks-20261015 checks
python3 queue_screened_prefill_reset_profiles.py ROOT ROOT/logs/screened-prefill-reset-native-20261015
python3 archive_screened_prefill_reset.py ROOT ARCHIVE
python -X utf8 verify_screened_prefill_reset.py EVIDENCE VERIFICATION_JSON
python3 launch_confirm_screened_prefill_reset.py ROOT ROOT/logs/screened-prefill-reset-confirm-native-20261015
python3 archive_screened_prefill_reset.py ROOT ARCHIVE screened-prefill-reset-native-20261015 screened-prefill-reset-confirm-native-20261015
python -X utf8 verify_screened_prefill_reset.py EVIDENCE CONFIRMATION_JSON screened-prefill-reset-native-20261015 screened-prefill-reset-confirm-native-20261015
```
