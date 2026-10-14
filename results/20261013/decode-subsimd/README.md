# Sixteen-lane decode groups: full-model integration rejected

The broad 64-thread integration is rejected and its five serving/worker files
are restored to their verified originals on Windows and the isolated Mac.
All 19 validation stages and 72 complete native outputs pass, but decode is
slower in every full-model profile. The repeated component gains below did
not carry into serving. No favorable-number repeat or fresh vLLM comparison
is justified for this rejected integration; the full goal remains unmet.

## Actual native request results

Native ABBA uses two fresh workers per build/profile, three requests each,
with four warm samples after excluding each worker's first request. Same
checkpoint values, precision, prompts, capacities, caps, stopping rules,
image bytes and chunk plans; fresh image encoding and sequential execution.

| Model | Batch | Control TPS/row | Candidate TPS/row | Decode ratio | Control TTFT ms | Candidate TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 58.05 | 57.81 | 0.9960 | 51.89 | 52.89 |
| Qwen3-0.6B | 2 | 57.57 | 56.83 | 0.9871 | 86.09 | 87.31 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 71.77 | 68.45 | 0.9538 | 36.21 | 35.79 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.84 | 67.33 | 0.9504 | 56.62 | 57.84 |
| Qwen3.5-0.8B | 1 | 47.78 | 47.21 | 0.9882 | 218.43 | 216.67 |
| Qwen3.5-0.8B | 2 | 44.57 | 43.86 | 0.9843 | 383.28 | 384.77 |

Coder loses about 4.6–5.0% decode, while text/VL lose about 0.4–1.6%.
Small coder/VL B1 TTFT improvements do not justify the throughput regression.
No cause for the component/serving difference is established. Aggregate TPS
at B2 is twice the displayed per-row rate for these equal rows. First-use
TTFT after setup is mixed; all separate means are in
[native/summary.json](native/summary.json). Initialization is also separately
recorded, and peak RSS changes are small/mixed (-3.1 to +1.1 MiB). No memory
saving is established. OS/driver caches were not flushed; separate intervals
are not summed and labeled an observed hardware-cold TTFT.

All numerical/serving/head/vision checks retain their previous bounds,
including 48 exact-tail comparisons, 24 private-cache profiles and 36 extended
VL comparisons. Complete prompt/output IDs, stopping flags and chunk plans
match across all 72 requests. Native evidence, source/build/checkpoint hashes,
actual feature flags, guards and exits are in [native/evidence](native/evidence/).

Two actual component runs pass 272 comparisons on real first-layer matrices,
complete vocabulary heads and irregular shapes from all three checkpoints,
at batches 1/2. Both candidates preserve BF16 learned values and FP32
input/accumulation/output. Reduction order changes: results are not generally
bit-identical. No numerical bound is widened.

The candidate gives each 32-lane SIMD group two independent 16-lane output
groups. Each lane holds the original two 32-way partial sums, combines them,
then reduces within its 16 lanes using XOR shuffles. No weight repacking,
additional weight allocation, prefix reuse or speculative decoding is used.
The measured variants have 64/128 threads; the native candidate uses 64.

## Component results

Ratios below pool control GPU ms / candidate GPU ms across collected matrices
with more than 256 outputs, once per matrix/batch. They are unweighted by
model layer counts and cannot be treated as decode TPS improvements.

| Model | 64 threads initial / repeat | 128 threads initial / repeat |
| --- | ---: | ---: |
| Qwen3-0.6B | 1.0254 / 1.0195 | 1.0179 / 1.0102 |
| Qwen2.5-Coder-0.5B-Instruct | 1.0276 / 1.0137 | 0.9976 / 1.0171 |
| Qwen3.5-0.8B | 1.0526 / 1.0303 | 1.0354 / 1.0372 |

The initial 64-thread batch-1 head ratios are 1.0556/1.0814/1.0957
for text/coder/VL; the repeat is 1.0288/1.0235/1.0291. This contraction is a
reason to require full-model measurements, not infer the larger initial gains.
The repeat's batch-2 head ratios are 1.0223/1.0202/1.0414. Full individual
shapes and timings remain in raw summaries and [component-pooled.json](component-pooled.json).

Each shape/variant uses two ABBA rounds, three warmups per position and
twelve measured operations per position. GPU times are component command
durations; CPU wall timings are also retained. Checks cover input/output
strides, output padding and odd/aligned inner/output tails. Every candidate
passes `5e-6 + 5e-5*abs(control)` and an independent FP64 envelope:
`gamma(ceil(K/32)+32)*sum(abs(products)) + 1e-7`, with float epsilon.
The control independently passes that envelope too. Worst candidate FP64
ratio is 0.00997263774 in both runs. This checks sampled matrix inputs;
it does not prove complete model generation or all future states.

## Tested integration, now reverted

The tested `GARNET_METAL_SUBSIMD_DECODE` defaulted OFF and conflicted with the separate
experimental native-bfloat body option. Only BF16 one/two-row projections
with more than 256 outputs select the candidate. Prefill, FP32-weight paths
and the selected small-output kernel retain their existing routes. The new
pipeline requires 32 hardware lanes and at least 64 threads. The primary
library remained unchanged; a fresh `build-decode-subsimd` was used. The
build option, appended shader, selector and feature marker are now removed
from current serving sources. Reproduction of the rejected candidate must
use the archived candidate source/worker files; its binary is retained.

The integrated 64-thread shader matches the tested source exactly, SHA256:
`b762de547c14fbf567b68470e12522fb88a287334d61ff6cc893ed5ef37c1342`.
Eight source files were installed into the isolated Mac root after verifying
original/new identities and preserving originals. The
[native source manifest](native-source-manifest.json) records them.
Source bundle SHA256:
`576f3ae0265f45997a471cf82db37c3a685dec5cbc1033810db38af2e0a72bc9`.

The candidate configured, built, passed the native MPS reference CTest and
all remaining trained-model, cache, head, vision, exact-tail and extended-VL
checks. The six-profile native ABBA then completed with the regressions above.

Owned root: `/Users/zsdx/garnet-metal-investigation`. Checks wrapper PID 33098,
`logs/decode-subsimd-checks-20261013`; queued inference wrapper PID 35033,
`logs/decode-subsimd-inference-20261013`. Both are terminal exit 0, with no
live handle remaining. Their launchers retain exact absolute commands.
Native control is the unchanged attention-prefix library SHA256
`f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088`.
CPU-direct reset remains OFF on both sides to isolate the decode layout.
There is no second native timing run of this uniformly slower candidate.

## Reproduction and evidence

Component command: `run_decode_subsimd_probe.py ROOT NEW_OUTPUT_DIR`.
Both component jobs exited 0 (PIDs 31269 and 32059). Exact source snapshots,
generated shader/harness, commands, hardware, checkpoint/matrix hashes,
all quality/timing records and process monitors are in [evidence](evidence/).
Raw `.bin` matrices are omitted and retained privately, with 68 verified
size/hash records in `omitted-matrix-proof.json`; their source can regenerate
them from the pinned checkpoints. No matrix array was deleted by this work.

```text
python test2026/metal/verify_decode_subsimd.py results/20261013/decode-subsimd
```

The verifier checks 81 file hashes, both terminal runs, source identities,
272 numerical/padding checks, checkpoint-derived matrix proofs and timing
calculations. Component archive SHA256:
`dcf1f709c54350db4144f6cb78424de9d97672c0c26c22db14ecab2c4e5ed1b6`.
The [Metal language resources](https://developer.apple.com/metal/resources/)
provide the specification for the shuffle operations; actual support is
established here by compilation and execution on the M4.

The additional native archive has 417 verified files, 19 terminal stages and
72 complete matching outputs. Its SHA256 is
`60ea68ffd5353bf38426e03474e587affc8f90d086220f3a1502dda47934cae3`.
Rejected candidate library SHA256:
`9d051795c22c4f5d0df30329370204e0328fe5ce8338b5f20e3fc601dd8fba6e`.

```text
python test2026/metal/verify_decode_subsimd_native.py results/20261013/decode-subsimd/native
```

The native verifier checks raw hashes, original/installed/reference identities,
explicit validation counts, build identities, complete outputs, actual layout
flags with CPU-direct reset OFF on both sides, fresh frames, timing arithmetic
and all 24 monitored worker exits. The old template's scope/terminal message
uses a reset/BF16-body label; feature flags, executed source and library hashes
identify the actual tested layout. Restoration first verified every current
authored-file hash before replacing only five files with their preserved
originals. [Restoration hashes/script](restoration/) and the before/after
manifest retain that proof. Prior primary/attention/direct-reset libraries and
all component/native evidence and binaries were preserved.

The GPT-OSS/Pi checkouts and primary Metal binaries remain intact. The full
three-model 2x decode/half-TTFT goal remains unmet; the
[latest actual vLLM comparison](../selected-attention-vllm/README.md) retains
all six profiles and their remaining targets.
