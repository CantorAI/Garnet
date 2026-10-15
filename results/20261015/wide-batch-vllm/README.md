# Fresh matched comparison for the validated latency bundle

PID3095 exited0 in578.0169s. An independent audit verified530 archived files,
three actual GPU dtype audits and72 complete matching outputs from all six
sequential Garnet/vLLM/vLLM/Garnet profiles. This is one matched cohort; full
confirmation remains required. The full goal is unmet.

| Model | Batch | Garnet /vLLM tokens/s per row | Garnet /vLLM warm TTFT ms |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B |1 |63.5260 /37.7679 |37.0659 /74.2021 |
| Qwen3-0.6B |2 |61.4839 /28.0106 |58.8427 /89.4655 |
| Qwen2.5-Coder-0.5B-Instruct |1 |79.7349 /47.0249 |26.9086 /57.7704 |
| Qwen2.5-Coder-0.5B-Instruct |2 |75.3592 /28.3400 |42.1604 /84.5120 |
| Qwen3.5-0.8B VL |1 |53.7989 /25.8580 |178.3726 /257.4076 |
| Qwen3.5-0.8B VL |2 |50.4185 /17.2814 |301.9957 /429.8688 |

Four profiles reach2x decode and three reach half TTFT in this cohort. TextB1
TTFT fraction0.4995269 and coderB2 fraction0.4988691 are very near the threshold;
they are not robust repeated wins. Only coderB2 meets both thresholds in this
single cohort. Text/coderB1 decode and textB2/VL TTFT remain material gaps.
Aggregate B2 throughput is twice the listed per-row rate. Initialization and
first-request observations remain separate in raw records.

Archive SHA256:
`3b26eb57ca5d31cae0e8e2e995aaab355687ec967cddcfc29f59f0844b603533`.

The separately pinned selection file is
`work/wide-batch-next-comparator-selection.json`. It selects the validated
`build-wide-batch-prefill` library for Qwen3-0.6B, Qwen2.5-Coder-0.5B-Instruct
and Qwen3.5-0.8B VL, B1/B2. The existing selection files are preserved. The
native build includes screened head,1024-row seed, interleaved screening index,
ordered prefill streaming, dirty-prefix state reset, serial vision prefetch,
and the bounded160-row prefill admission. VL B2 now uses one80-token internal
chunk; the physical total remains80, with unchanged external prompt, image,
capacity, output cap, stops and expected complete generated tokens.

This is an additional matched-FP32 comparison. Learned BF16 checkpoint values
are expanded losslessly for the comparator, and fresh audits must establish
actual FP32 parameter, activation and KV execution on Metal. Native original
BF16/scaled-half storage uses different memory traffic and does not establish
a stock-BF16 comparator result. Both engines use the same image frontend.
Prefix reuse and speculation remain disabled. Runtimes execute sequentially
with process/resource guards. Init, first request after init and subsequent
warm requests are recorded separately; caches are not flushed and no hardware-
cold total is inferred.

Launch was gated on the independently archived original28 native stages and
two complete72-output native trials. The separate body-packing component is
not integrated into this comparison.

```text
python3 launch_wide_batch_vllm.py ROOT ROOT/logs/wide-batch-vllm-20261015
```

The full2x decode / half-TTFT goal remains unmet until complete repeated matched
measurements prove every agreed model/batch target.
