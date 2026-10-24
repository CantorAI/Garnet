# SGLang transport correction — 2026-10-23

The completed run `f948ed75a0af457199c313f349ba19e5` used `localhost`, whereas the archived vLLM run used literal `127.0.0.1`. Exclude this SGLang trial from matched speed ratios.

Paired TCP checks, without model requests, measured median connection times of 2.0612463 seconds for localhost and 0.0151026 seconds for IPv4. The resolved `::1` connection was refused after 2.0430977 seconds before IPv4 succeeded. Do not subtract these values from measured TTFT; rerun the actual workload.

The retained trial archive has 768260 bytes and 440 entries. SHA256: `ad4306a58ab79da35c948039cc98ac3d9b632248f803bc3c9704517780e53c0f`. Independent local verification checked CRC, every file hash, 64 measured SSE streams and eight recomputed summaries. Actual request bytes and token counts match archived vLLM. The transport guard correctly suppresses all eight ratios.

The controlled `sglang-corrected-ipv4-benchmark` rerun uses the same live SGLang server and workloads with literal IPv4. Results remain pending. No native GPU probe, source synchronization, rebuild or heavy artifact transfer may overlap its timing. The 2x goal remains unmet.
