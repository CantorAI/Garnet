# Global-alpha bundle versus vLLM-Metal

The full goal remains unmet. The actual six-profile comparison exited 0 and
independently verified 417 archived files, three real GPU dtype audits and 72
complete matching requests. All batch-2 profiles exceed 2x decode throughput
in this run. No batch-1 profile does, and no warm first-token latency reaches
half the comparator's latency.

| Model | Batch | Garnet tokens/s per row | vLLM tokens/s per row | Decode ratio | Garnet TTFT ms | vLLM TTFT ms | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 58.8102 | 37.7218 | 1.5591 | 48.9037 | 76.4773 | 0.6395 |
| Qwen3-0.6B | 2 | 58.0651 | 25.8420 | 2.2469 | 80.9695 | 95.0266 | 0.8521 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 72.2365 | 46.6727 | 1.5477 | 33.6828 | 55.2283 | 0.6099 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 70.5367 | 28.4020 | 2.4835 | 52.0362 | 84.2238 | 0.6178 |
| Qwen3.5-0.8B VL | 1 | 48.1185 | 26.3395 | 1.8269 | 198.6298 | 258.0453 | 0.7697 |
| Qwen3.5-0.8B VL | 2 | 44.7354 | 16.9896 | 2.6331 | 359.0334 | 438.6902 | 0.8184 |

Four warm requests per runtime/profile from two fresh sequential workers.
Each worker's first request is excluded. B2 aggregate throughput is twice the
per-row rate. Decode excludes the first prediction and includes terminal
predictions. TTFT includes request preparation and prefill. Full precision
targets, aggregate throughput, initialization and first-use measurements are
in `remaining-budgets.json`; GPU and wait intervals overlap and must not be added.

Relative to these observed means, batch-1 decode still needs another 28.3%
text, 29.2% coder and 9.5% VL improvement. First-token latency needs another
18.0–41.3% reduction across the six profiles. Comparator fluctuations between
days must not be attributed to the native change; the separate native ABBA
trials establish the repeated 4.7–11.6% warm latency direction.

Initialization and first request AFTER initialization remain separate observations:

| Model / batch | Garnet init s | vLLM init s | Garnet first-request TTFT ms | vLLM first-request TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Text / 1 | 1.7505 | 16.9680 | 157.20 | 67.99 |
| Text / 2 | 2.0840 | 16.4049 | 187.99 | 108.24 |
| Coder / 1 | 1.3992 | 15.8567 | 125.52 | 60.76 |
| Coder / 2 | 1.3905 | 15.9328 | 144.99 | 98.26 |
| VL / 1 | 4.0589 | 33.6894 | 349.90 | 310.66 |
| VL / 2 | 4.1203 | 33.2868 | 571.64 | 538.81 |

Native initialization is faster, but first-request TTFT after initialization
is slower in all six profiles. OS/driver caches were not flushed. No sum is
presented as an observed hardware-cold end-to-end latency.

This is the additional matched-FP32 study, not a stock BF16 win claim. Learned
values restore exactly; arithmetic, activation and KV precision match FP32.
Native BF16/lossless scaled-half weight storage and losslessly expanded FP32
comparator storage have different widths and bandwidth traffic. Prompts,
images, resolution, caps and B1/B2 profiles remain frozen. Prefix caching and
speculation are disabled, all images are freshly encoded and all complete
normalized token sequences, prompt IDs and finished flags match. Actual
global-alpha counters are positive and unrelated experimental features are off.

The M4 Mac mini has 16 GiB RAM and macOS 15.6.1. Recorded isolated comparator
versions: vLLM 0.30.0+cpu, vLLM-Metal 0.30.0, MLX 0.32.1, mlx-lm 0.32.0,
mlx-vlm 0.6.17, Torch 2.13.0, Transformers 5.19.0, NumPy 2.3.5, Pillow 12.3.0.
The real dtype audits confirm MLX GPU execution with FP32 parameters,
activations and KV; the vLLM package's `+cpu` version string does not by itself
describe the Metal worker's device.

Owned PID 40956 completed at `ROOT/logs/global-alpha-vllm-20261014`. Guards
sample every 0.5 seconds, exempt only the owned process group, reject other
inference/probe jobs and check at least 2 GiB disk free. No competing runtime
is intentionally run simultaneously; arbitrarily short activity between
samples cannot be categorically excluded. The original library selection,
GPT-OSS checkout and Pi checkout are preserved.

Archive SHA-256:
`950102f920b0facd61d45a4063233fb6a8e95fa42f64f6ce97e310c4ee37f580`.
The frozen selection is `selection.json`; the full archive manifest,
checkpoint equivalence proofs, commands, exact executed scripts, monitor logs,
dtype observations and versions are in `evidence/`. Local audit is
`verification.json`. A second full run alone would not resolve the missing
targets, so further optimization is required.

```text
python3 run_global_alpha_vllm.py ROOT ROOT/logs/global-alpha-vllm-20261014/jobs
python3 archive_global_alpha_vllm.py ROOT ARCHIVE
python verify_global_alpha_vllm.py RESULTS VERIFICATION_JSON
python summarize_global_alpha_vllm.py RESULTS
python summarize_global_alpha_phases.py RESULTS
```
