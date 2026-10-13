# vLLM-Metal comparison evidence

Target: Garnet >=2x vLLM-Metal decode throughput and <=0.5x its first-token
latency on the same M4 16 GiB Mac and identical BF16 checkpoints. The first
clean matched baseline executed 48 requests. Garnet is slower in all four
profiles; neither target is met.

## Clean baseline

`baseline/vllm-matched-clean-baseline-20261012/jobs/summary.json` is the
authoritative repeated baseline for native build `d2a7146`. Each row below
averages four warm requests per runtime, following first-use requests in two
fresh processes. Full first-use results, generated token IDs, commands,
individual engine steps, process snapshots and model/library hashes are
retained in that folder. All 48 outputs and prompt token IDs match exactly;
all processes shut down cleanly.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 37.62 | 73.83 | 150.22 | 52.07 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 62.11 | 92.24 | 96.79 | 34.77 |
| Qwen3-0.6B | 2 | 27.20 | 75.34 | 280.62 | 71.12 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 55.83 | 92.36 | 157.85 | 38.03 |

Hardware: Apple M4 Mac mini, 16 GiB, macOS 15.6.1. Comparator: isolated
CPython 3.12.15, vLLM 0.30.0+cpu, vLLM-Metal 0.30.0, MLX 0.32.1,
mlx-lm at `9e6acca691e64d6d8bb808c328fcdea459099cca`. Despite the vLLM wheel's
`+cpu` label, the Metal plugin and MLX GPU execution are active and the device
probe reports Apple M4. Both runtimes use the same BF16 checkpoint files.
Garnet uses FP32 activations/KV/accumulation; vLLM uses BF16 model/KV execution.
This compares checkpoint/weight precision, not identical internal arithmetic.

The frozen workload has capacity 256, greedy generation, output cap 64,
no prefix reuse or speculation, identical native prompt IDs, and batch 1/2.
Decode speed uses the complete first-to-final prediction wall interval.
Per-request reset/cache preparation is included in TTFT. Filesystem caches
are not flushed. RSS is individual-process high water, not aggregate worker
or unique GPU memory. Two existing WorldSense service processes were verified
to have no inference/GPU libraries and left running; their CPU/RSS snapshots
are saved. Competing inference runtimes never overlap.

## Visible-cache optimization

Attention now scores and accumulates only the causal visible prefix. Cache
storage, slot bounds, state reset, weight precision and context capacity stay
the same. The inactive-row controls retain their previous fallback behavior.
All 11 numerical/serving/vision validation jobs passed, followed by another
48-request matched suite with identical outputs and clean shutdown. Evidence:
`visible-cache/visible-cache-checks-20261012` and
`visible-cache/vllm-matched-visible-cache-20261012/jobs/summary.json`.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Garnet TTFT, ms | vLLM TTFT, ms |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1 | 40.18 | 74.40 | 118.77 | 51.51 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 67.73 | 92.33 | 75.40 | 34.72 |
| Qwen3-0.6B | 2 | 28.49 | 75.53 | 214.42 | 70.55 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 61.01 | 92.00 | 113.86 | 38.30 |

Against the native baseline, decode improves 4.8–9.0% and TTFT falls
20.9–27.9%. Neither vLLM-relative target is met. GPU time remains the principal
decode cost; these gains are not evidence of a 2x comparator advantage.

## Comparator teardown compatibility

The preceding 48-request suite in
`baseline/vllm-matched-native-syntax-baseline-20261012` generated identical
outputs but its vLLM workers crashed during shutdown. Its derived summary
flags `clean_backend_shutdown=false`; a parent exit code of zero does not
make that suite clean. Raw failures remain intact.

The private comparator patch skips Torch accelerator cache cleanup for
platforms whose Torch device type is CPU. The Metal plugin uses that metadata
for compatibility while inference runs on MLX. The original vLLM cleanup
checked only its OOT platform enum and entered the Torch accelerator cleanup
path incorrectly. No inference, scheduler or numerical code was changed.
`baseline/vllm-cleanup-compatibility-20261012` contains the exact one-condition
diff, original/patched SHA-256 and provenance; the second full suite proves
clean shutdown. `patch_vllm_metal_cleanup.py` verifies the official wheel's
original source hash and atomically patches only this isolated installation.

`setup/` retains completed setup attempts and the verified wheel-transfer job.
These are installation/network diagnostics, excluded from all inference timing.
The cached-URL setup was interrupted deliberately after verified local wheels
became available; its separate `interruption.json` records that action. A retry
with an incomplete UV copy failed the size/digest guard before installation.

`baseline/` retains the later setup, download, compatibility and benchmark
attempts, including interrupted transfers and early guard/parser failures.
The successful offline installation resolved 161 packages. Verified public
wheels, requirements and installed versions are recorded in the manifests.
All setup/download time is excluded from inference measurements.

The runner files have now passed actual native and vLLM engine execution. See
`test2026/metal/VLLM_COMPARISON.md` for matching, accounting and precision limits.
The GPT-OSS and Pi checkouts were checked clean and were not edited.
