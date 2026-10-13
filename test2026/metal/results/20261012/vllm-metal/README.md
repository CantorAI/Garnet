# vLLM-Metal comparison evidence

Target: Garnet >=2x vLLM-Metal decode throughput and <=0.5x its first-token
latency on the same M4 16 GiB Mac and identical BF16 checkpoints. No matched
vLLM-Metal inference result has been recorded in this evidence snapshot yet.
The target remains unproven.

`setup/` retains completed setup attempts and the verified wheel-transfer job.
These are installation/network diagnostics, excluded from all inference timing.
The cached-URL setup was interrupted deliberately after verified local wheels
became available; its separate `interruption.json` records that action. A retry
with an incomplete UV copy failed the size/digest guard before installation.

As of 2026-10-13 02:00 UTC, the isolated local-wheel setup remained active and
was downloading required PyPI dependencies. Its owned job is
`logs/vllm-setup-local-wheels-20261012` on the Mac. The durable job
`logs/vllm-matched-baseline-20261012` waits for successful setup, checks actual
imports and the Metal device, then executes and summarizes the frozen comparison
sequentially. Setup and benchmark output must be collected after completion;
queued work is not evidence of executed inference or a speed improvement.

The prepared runner files passed Python syntax checks and `git diff --check`.
They have not yet passed an actual vLLM engine execution. See
`test2026/metal/VLLM_COMPARISON.md` for matching, accounting and precision limits.
The GPT-OSS and Pi checkouts were checked clean and were not edited.
