# Direct resident-state reset: repeated latency gains and a decode tradeoff

The optional direct reset removes temporary zero-filled CPU vectors and their
copies into shared Metal state buffers. Two complete native ABBA runs preserve
144 full output sequences, and all 19 validation stages pass. Warm text TTFT
improves in both runs, but text batch-2 decode is 1.7–2.4% slower. The full
2x-vLLM/half-TTFT goal remains unmet; this is not a new vLLM comparison.

The `GARNET_METAL_DIRECT_STATE_ZERO` option defaults OFF. The candidate clears
logical float elements in existing buffers, preserving row padding, positive
zero, allocation identity, readiness rules and the closed-command requirement.
Decode arithmetic, checkpoint precision and graph scheduling are unchanged.
The primary serving library and GPT-OSS/Pi checkouts are untouched.

## Actual full-request results

Four warm requests per build/profile per run, from two fresh workers with
three requests each. First requests are excluded. Both builds have the same
attention-prefix policy; direct reset is the tested difference. Runtimes run
sequentially under the existing conflict guards. Prompts, stop rules, output
caps, chunk plans and actual image bytes match; images are freshly encoded.

| Model | Batch | Initial TTFT control → candidate ms | Initial reduction | Confirmation TTFT control → candidate ms | Confirmation reduction | Decode candidate/control initial / confirmation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 52.90 → 50.35 | 4.83% | 53.67 → 48.91 | 8.86% | 1.0038 / 0.9900 |
| Qwen3-0.6B | 2 | 87.38 → 75.95 | 13.08% | 87.38 → 73.20 | 16.22% | 0.9762 / 0.9827 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 36.10 → 36.18 | -0.22% | 35.89 → 34.88 | 2.81% | 1.0077 / 1.0004 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 56.85 → 55.80 | 1.84% | 56.78 → 55.09 | 2.97% | 1.0003 / 0.9954 |
| Qwen3.5-0.8B VL | 1 | 216.11 → 218.19 | -0.96% | 216.46 → 213.25 | 1.48% | 1.0212 / 1.0044 |
| Qwen3.5-0.8B VL | 2 | 380.23 → 376.64 | 0.94% | 379.81 → 377.50 | 0.61% | 1.0064 / 0.9959 |

Text TTFT gains repeat, and coder B2 has a smaller repeated signal. VL B2's
sub-1% signal is too small to establish a practically useful gain here.
Coder B1 and VL B1 warm gains do not repeat. No decode improvement is proven;
text B2's repeated slowdown is a material tradeoff, without an established
cause. Retain the option for investigation, with no uniform serving promotion.
Do not mix these candidate timings with an older vLLM run to claim new ratios.

## Reset, first use and memory

Reset remains inside TTFT on both sides. Confirmation warm reset means:

| Model | Batch | Control reset ms | Direct reset ms |
| --- | ---: | ---: | ---: |
| Text | 1 | 7.207 | 2.975 |
| Text | 2 | 23.676 | 5.096 |
| Coder | 1 | 0.863 | 0.417 |
| Coder | 2 | 1.736 | 0.708 |
| VL | 1 | 3.900 | 1.601 |
| VL | 2 | 11.014 | 3.660 |

All reset means also fall in the initial run. This supports the intended
mechanism, but reset savings are not identical to full-request savings.
The existing VL `vision_seconds` interval includes reset; the new separate
`reset_seconds` field permits distinguishing it without changing TTFT scope.

First-request TTFT after initialization improves in both runs for text/VL;
coder B1 is slower in both, and coder B2 is mixed. Confirmation first-request
means in milliseconds (control → candidate): text B1 155.44 → 149.71,
text B2 187.99 → 170.20, coder B1 122.28 → 125.43, coder B2 136.19 → 142.88,
VL B1 350.43 → 334.25, VL B2 565.69 → 551.97. Initialization is separately
recorded in [summary.json](summary.json). OS/driver caches were not flushed;
no sum is presented as an observed hardware-cold end-to-end TTFT.

Although temporary allocations are removed, measured peak RSS changes are
small/mixed (about -14.7 to +7.8 MiB across both runs); no peak-memory saving
is established. The largest confirmation text reset still takes 5.1 ms.
A future GPU buffer-fill reset could be measured against this CPU clear;
that method is not implemented or benchmarked by this evidence.

## Validation and evidence

Actual M4 build and both CTest references pass. The direct-zero test checks
36 repeated nonzero-to-positive-zero transitions, padded shapes and actual
GPU consumption, plus empty-buffer and open-command rejection. The remaining
stages include 48 exact-tail comparisons, 24 private-cache profiles with
session isolation/reset checks, standard serving/image/trained-layer/head/
vision suites and 36 extended Qwen3.5 prefill comparisons. Existing numerical
bounds are unchanged. Two native ABBA runs cover all six model/batch profiles
with 144 matching complete token/prompt sequences, stopping flags and plans.

Owned root: `/Users/zsdx/garnet-metal-investigation`. Original files were
preserved before installing ten sources with verified identities. The local
[source manifest](source-manifest.json) and archived originals retain the
before/after hashes. All durable jobs are terminal exit 0: checks PID 6169,
initial inference PID 8442, confirmation PID 27472. No live job was restarted
because an observation timed out; no unrelated process was stopped.

Exact absolute commands, installed/generated runner sources, reference
scripts, build flags, library/model hashes, monitors, timings, reset telemetry,
complete outputs and exits are in [evidence](evidence/). The executed comparator
inherits an older template's BF16-body scope label; actual build hashes,
`direct_state_zero_enabled` and `attention_prefix_enabled` fields identify the
reset experiment. That label does not alter the executed work or its scope.

```text
python test2026/metal/verify_direct_state_zero.py results/20261013/direct-state-zero
```

The verifier checks 677 file hashes, installed/original/reference identities,
19 terminal validation stages, explicit check counts, both library identities,
144 complete outputs, runtime flags, fresh frames, timing arithmetic and all
48 monitored worker exits. Raw bytes are preserved by `.gitattributes`.

- Candidate library SHA256: `b33ee094e60efb1fa57de7b6141cb2b2650509f44cd218c77844b3b94c2ec451`.
- Native comparison control: `f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088`.
- Unchanged primary library: `73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
- Source bundle: `8941aa256fc62380accf5db9ad90ffa0da0bff206d742a055e657a9712e4a710`.
- Evidence archive: `82ed28a3c4e45c1fd00ff0980239bffe4a64584cb1545e6ee2ac5788b530e008`.

VL B2's attention-prefix control here isolates reset and does not promote
the otherwise inconclusive attention-prefix VL B2 policy. The latest
[actual vLLM comparison](../selected-attention-vllm/README.md) remains the
authoritative six-profile comparator result, with all unmet targets recorded.
