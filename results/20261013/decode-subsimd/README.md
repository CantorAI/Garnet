# Sixteen-lane decode groups: component gains, native validation pending

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

## Optional native integration

`GARNET_METAL_SUBSIMD_DECODE` defaults OFF and conflicts with the separate
experimental native-bfloat body option. Only BF16 one/two-row projections
with more than 256 outputs select the candidate. Prefill, FP32-weight paths
and the selected small-output kernel retain their existing routes. The new
pipeline requires 32 hardware lanes and at least 64 threads. The primary
library remains unchanged; a fresh `build-decode-subsimd` is used.

The integrated 64-thread shader matches the tested source exactly, SHA256:
`b762de547c14fbf567b68470e12522fb88a287334d61ff6cc893ed5ef37c1342`.
Eight source files were installed into the isolated Mac root after verifying
original/new identities and preserving originals. The
[native source manifest](native-source-manifest.json) records them.
Source bundle SHA256:
`576f3ae0265f45997a471cf82db37c3a685dec5cbc1033810db38af2e0a72bc9`.

The candidate configured, built and passed the native MPS reference CTest.
The remaining trained-model, cache, head, vision, exact-tail and extended-VL
checks are running. A full six-profile native ABBA comparison is queued behind
their successful terminal result. No whole-model benefit or new vLLM ratio
is claimed while those results are pending.

Owned root: `/Users/zsdx/garnet-metal-investigation`. Checks wrapper PID 33098,
`logs/decode-subsimd-checks-20261013`; queued inference wrapper PID 35033,
`logs/decode-subsimd-inference-20261013`. Both were last verified live. Inspect
their actual processes and `exit.json` before continuing; do not restart on
an observation timeout. Their launchers retain exact absolute commands.
Native control is the unchanged attention-prefix library SHA256
`f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088`.
CPU-direct reset remains OFF on both sides to isolate the decode layout.
Complete output equality and repeat measurements remain required.

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

The GPT-OSS/Pi checkouts and primary Metal binaries remain intact. The full
three-model 2x decode/half-TTFT goal remains unmet; the
[latest actual vLLM comparison](../selected-attention-vllm/README.md) retains
all six profiles and their remaining targets.
