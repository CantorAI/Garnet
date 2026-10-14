# Native BF16 body decode: small repeated gains, latency tradeoff

An optional `GARNET_METAL_NATIVE_BFLOAT_BODY=ON` route loads original BF16
weight bytes through Metal's native `bfloat` type. One-row body projections
use the original serial accumulation with 64 threads; two-row body projections
use vector dot products with 128 threads. FP32 inputs, accumulation and output
remain. The vector-dot route reassociates sums; learned values are not rounded.
There is no weight repacking, prefix/encoder reuse or speculative decoding.

Admission is bounded to one/two rows, input widths divisible by four and
257 through 99,999 output columns. Small outputs and the complete vocabulary
heads keep their selected kernels. Prefill dispatch is unchanged. Native BF16
source and Metal language version 3.1 are enabled only by the opt-in build
flag. The default remains OFF and does not compile the extra kernels. Actual
compatibility/performance here is established on this M4, not all Mac GPUs.

## Component and numerical evidence

The initial real-weight component run covers four variants (serial/dot,
64/128 threads), first-layer matrices, all three complete heads, batches 1/2,
irregular input widths and padded outputs: 272 checks. A confirmation tests
serial/64 and dot/128 in 136 checks. All 408 pass the independent FP64 dot
reference, with fixed gamma(n)*sum(abs(products))+1e-7 bound and
n=ceil(inner/32)+32 using float epsilon. Worst bound ratio is 0.00997263774.
The control must pass too. Serial candidates are bit-identical; dot candidates
can differ due to summation order. No existing trained-layer bound is changed.

Initial heads mostly regress, so they are excluded from serving admission.
Large body component sum ratios repeat at roughly 1.03–1.05 for selected text
variants and 1.05 for VL batch-2 dot/128. These sums cover collected matrices
once per batch and are not weighted by layer counts or model tokens/sec.
Coder component signals are small/mixed.

The integrated build passes all eleven ordinary stages (native kernel test,
serving, mixed images, trained Qwen3.5/dense layers, all three complete heads,
small/pretrained vision), plus three private-cache suites covering 24 profiles.
Existing numerical thresholds and complete greedy output checks remain.
An additional actual GPU test verifies native BF16-to-FP32 casts for all
65,280 finite patterns, including subnormals and signed zero, under precise
and fast-math compilation: zero bit mismatches in both modes. Nonfinite
weights retain existing upload rejection.

The SDK exposes `MPSDataTypeBFloat16`, but earlier actual direct-BF16 MPS
execution on this Mac aborted. That compatibility test was not repeated;
the working exact-half prefill route remains. Native shader BF16 loads are
distinct from direct BF16 MPS matrix multiplication.

## Actual native inference results

Same M4 / 16 GiB, checkpoints, prompts, image bytes, capacities, output caps,
stop rules and retained prefill plans. Each native comparison uses ABBA build
order, two fresh persistent workers per build, three requests per worker and
four non-first requests per warm mean. Every complete sequence, prompt ID,
finished flag and VL fresh-image identity matches the frozen baseline.
There are 72 initial plus 48 confirmation requests. No runtimes run together.

| Model | Batch | Initial decode ratio | Confirmation decode ratio | Confirmation TTFT fraction |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 1.01675 | 1.01016 | 1.01025 |
| Qwen3-0.6B | 2 | 1.01644 | 1.02801 | 1.02577 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.99890 | Not repeated | Not repeated |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.99550 | Not repeated | Not repeated |
| Qwen3.5-0.8B | 1 | 1.00874 | 1.00488 | 1.00532 |
| Qwen3.5-0.8B | 2 | 1.01866 | 1.01588 | 1.00748 |

Ratios are candidate/control. The text decode direction repeats, about
1.0–2.8%; VL is about 0.5–1.6% in confirmation. Coder is flat and was not
repeated to seek favorable noise. Confirmation TTFT is slightly worse:
53.73 -> 54.29 ms and 90.11 -> 92.43 ms for text; 217.14 -> 218.29 ms and
387.24 -> 390.14 ms for VL. No latency benefit is established or attributed
to the decode algorithm. These are modest warm means, not a broad task suite.

The code is retained as a default-OFF experimental option. It is not selected
as the primary serving build, given the small throughput gain and latency
tradeoff. The selected control library stays byte-identical. There is no new
vLLM comparator run, stock BF16 advantage, or 2x/half-TTFT success claim.
The broader all-model goal remains active and unmet.

## Commands, identities and reproduction

```text
python3 run_native_bfloat_decode_probe.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_native_bfloat_confirm_probe.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_native_bfloat_body_checks.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_native_bfloat_body_profiles.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_native_bfloat_cast.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_native_bfloat_body_profiles.py INVESTIGATION_ROOT OUTPUT_DIR --confirm
python3 archive_native_bfloat_body.py INVESTIGATION_ROOT ARCHIVE_PATH
python3 verify_native_bfloat_body_evidence.py EVIDENCE_DIR PROOF_JSON
```

All six durable jobs exit zero. The integrated build/check runner records 16
successful configure/build/check stages. The comparison and cast were gated
on successful predecessors. Monitors reject competing runtimes/probes every
0.5 seconds, exempt only the created worker group, and terminate only owned
processes on failure. Arbitrarily short jobs between samples cannot be ruled
out. No interruptions or discarded attempts occurred in the selected runs.

The initial source bundle retains the six-profile manager before the later
`--confirm` change to select the four text/VL profiles. Exact commands, versions,
source/checkpoint/library hashes, raw ABBA windows, numerical logs, complete
outputs, startup/first-use and process monitors are in `evidence/`. Initialization
and first-use are separate; OS/driver caches were not flushed. No memory gain
is claimed. Checkpoint-derived matrices, raw RGB/patch assets and binaries are
omitted with sizes/hashes retained. A reproduction uses a fresh build/output
directory; the installer requires the archived original two source files and
the verified candidate bundle, rather than overwriting an arbitrary checkout.
GPT-OSS, Pi and the Mac original checkout remain untouched.

Candidate library SHA256:
`de330166f5592e6453444ee0b9fad66441f941f81f9f4a6b662ab7901b1346b5`.
Selected control SHA256:
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
Evidence archive SHA256:
`a5ba0d155270d69627a719e4b13a231b0151581d246ab7ba29224faa88607c02`.
