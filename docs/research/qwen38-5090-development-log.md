# Qwen3.8 RTX5090 research log

## Record policy

Append dated experiment entries; do not rewrite failed experiments into successes.
Use one fixed remote root `D:\CantorAI\5090-study-20261023`, one shared model cache,
and a flat `results` directory. Reuse build directories; keep accepted control
binaries and hashes separately from candidate output. Remove transfer archives
only after verified extraction. Keep compact raw JSONL and paper evidence; prune
rebuildable engines only when they are not needed by an active comparison.

For every future optimization record: experiment ID/time in UTC and local zone;
hypothesis; exact source diff and SHA-256; build flags/toolchain; model revision,
quantization, calibration and kernel dispatch; workload IDs and input hashes;
prompt/image token counts, output tokens, seeds, thinking/cache settings;
batch and client concurrency; control/candidate order; warmups and repetitions;
correctness and quality results; all raw latency/token/memory observations;
median/p95 and uncertainty; profiler configuration/overhead; failures;
interpretation, decision and limitations. Record power/temperature/clocks and
background load. Retain rejected candidates and interrupted runs as such.

Measurements, shape/memory estimates, hypotheses, and vendor claims are separate
evidence classes. Synthetic unit fixtures are not benchmark observations. GPU
profiling attribution is not production serving latency. No cherry-picked cohort
or changed precision/workload may silently support a 2x speedup claim.

## P000 — 2026-10-23: source and compatibility audit

Method: inspect the current Garnet working tree, retrieve pinned primary model
configs, compare architecture/quantization contracts and query target hardware
over SSH. The referenced chat exposed only message placeholders for prior
assistant claims, so no prior predicted token/s or TTFT is treated as evidence.

Source HEAD: `5b0dc7ff8e5c1a77807138db310605bcd12aa390`.
The original branch is `feature/gpt-oss-120b`, with substantial ongoing Qwen3.5
edits. A separate local worktree at `D:\CantorAI\work\qwen38-5090` uses branch
`qwen38-5090`. Current source was preserved through a per-file hashed snapshot;
its pre-existing dirty baseline is not attributed to this task or silently
committed. Future runs must retain both HEAD and complete source hashes.

Base source archive SHA-256:
`7c29e60a515827464a7fb0fa461d45bc0d5a91f93f6c00c7b7e476a1f8be9d99`.
Remote extraction verified that digest. `source-manifest.json` lists captured
Garnet/XLang3 files and exclusions; `preparation-manifest.json` supersedes
preparation files changed after the base snapshot. No credentials are included.

Observations: Qwen3.8-27B exists and uses Qwen3.5-family hybrid VLM architecture.
The 27B decoder, untied head and larger vision tower differ from the admitted
2B/4B models. NVIDIA's checkpoint mixes NVFP4 and FP8. A compressed-tensors
group128 symmetric W4A16 candidate is pinned, but not runtime-admitted. See the
implementation task and checkpoint evidence for exact revision/config hashes.

Target observation: Windows `zsdx-5090`, one RTX5090, 32607 MiB, driver617.14,
CUDA toolkit12.8. Existing `D:\CantorAI\Garnet` is deployment content, not a
buildable checkout. Preserved it and staged source under the fixed study root.
No target model weights were downloaded during preparation.

Outcome: 27B CUDA serving admission and packed NVFP4/FP8 integration are pending.
Current BF16 external projections and captured INT4 dequantization cannot be
treated as a working ModelOpt loader. No performance result or speedup exists.

## P001 — 2026-10-23: CPU preparation validation

Method: run `python -m unittest discover -s test2026/cuda/qwen38_5090 -p test_*.py -v`
locally with Python3.12.10 and remotely with `C:\Python312\python.exe` on
CD-RTX5090. Run offline preflight on the pinned NVIDIA config on both hosts.

Outcome: all9 tests passed on each host. Evidence-integrity checks validate
checkpoint-config hashes; negative tests reject incompatible model shapes and
bad timing records. Synthetic metric fixtures test correct decode token counts,
cohort throughput denominator, failed-request rejection and comparison against
the faster of both engines. These are CPU tests, not GPU inference validation.

Calculated B1/context4096 canonical-state sizes: attention KV536870912 bytes,
FP32 recurrent matrices150994944 bytes, convolution history7864320 bytes.
These omit weights, graphs, duplicated state, vision, activations and workspace;
they do not establish peak VRAM or prove fit on32GB.

## P002 — 2026-10-23: environment preparation

Method: start native CMake/Ninja/Visual Studio2022 Build Tools installation with
logs in the fixed remote root. User authorized WSL installation for reference
baselines. Installed WSL3.0.3 and enabled VirtualMachinePlatform without reboot.
Windows reports a restart is required before changes take effect. Native build
tools were still downloading at this entry; installation success is unverified.
Garnet remains a native Windows build. WSL is reserved for reference engines.

Outcome: baseline engines, native Release build, target GPU correctness,
quantized model fit and timed measurements remain pending. Do not publish
preparation timings as TTFT or claim the >=2x target is achieved.
