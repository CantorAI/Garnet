# Fresh lossless-body comparison against vLLM-Metal: running

Owned PID23200 is running three actual GPU dtype audits followed by six
sequential matched profiles. Launch requires two audited native trials with144
complete matching outputs, frozen original sources/library identities and
repeated decode gains across all six profiles. Selection is recorded at
`ROOT/work/coalesced-body-next-comparator-selection.json` and fixes the same
candidate library for all model/batch profiles.

Garnet candidate SHA256:
`c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1`.
The comparison retains identical learned values, FP32 arithmetic, prompts,
output limits, stop rules and image preprocessing. Comparator weights expand
losslessly to FP32. Garnet retains exact BF16 representations and additional
buffers. This is not a stock BF16 vLLM comparison. No prefix/vision reuse or
speculation is enabled. Runtimes run sequentially with resource guards.

Each full profile uses G/V/V/G fresh workers, three requests per worker.
Initialization, first-after-init and four warm requests per policy remain
separate. OS/driver caches are not flushed. Extra candidate memory/init costs
must be reported. Actual encoded body/block counters are mandatory.

No fresh comparator speed ratio or target completion is established yet.
Do not change pinned sources, selected library or drivers until terminal and
archived. The overall2x decode / half-TTFT objective remains active and unmet.

```text
python3 launch_coalesced_body_vllm.py ROOT ROOT/logs/coalesced-body-vllm-20261015
# Only after successful terminal execution:
python3 archive_coalesced_body_vllm.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_coalesced_body_vllm.py RESULTS RESULTS/verification.json
```
