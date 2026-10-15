# Combined threaded WebP decode and exact striped resize

Two actual CPU trials reduce fresh image preparation by 37–39%, about 19–20 ms,
while preserving all resized RGB and FP32 patch bytes. The independent audit
covers 56 archived files, 1912 compatibility comparisons and 384 timed frozen
image preparations. These are CPU times, not model TTFT or decode gains.

| Batch | Initial selected → combined ms | Confirmation selected → combined ms |
| --- | ---: | ---: |
| 1 | 49.4648 → 30.2425 | 49.7787 → 30.2271 |
| 2 | 51.8288 → 32.6303 | 51.9895 → 32.3147 |

The optional frontend composes the previously tested conservative static-WebP
decoder with the restricted exact-coordinate Pillow strip resizer. Each request
reads and decodes the original source anew. Metadata, animation and unsupported
WebP containers fall back to Pillow decoding; ineligible resize geometries fall
back to the original resize. The frozen image takes both accelerated routes.
No cached image pixels, embeddings, prefix state or speculative decoding is used.

Each compatibility trial covers 239 fixtures and 956 comparisons, including
lossless/lossy images, RGB/RGBA/L, EXIF orientations, animation and PNG/JPEG/BMP
fallbacks. There are 148 parallel resize calls and 808 resize fallbacks per trial.
Each timed batch has two ABBA rounds with 12 samples per window and common warmup.
Writes and hashes remain inside preparation timing. OS caches were not flushed.

The existing libwebp bridge SHA-256 is
`3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3`.
Pillow is pinned to 12.3.0 for the strip path. Input is 1372×1818 and resized
output is 192×288, width×height. GPU library and model data remain unchanged.

Owned CPU jobs terminated exit 0: PID 37557 took 135.0796 s including its wait
for the prior inference confirmation; PID 41308 took 74.7292 s. They did not
overlap inference. Combined archive SHA-256:
`3aa75dc363ec7339bcc47277b87a5d64343d115c724ac4135cfa4661d492a939`.

Actual image-plus-text inference started as owned PID 43673 at
`logs/threaded-striped-inference-20261015`, after both terminal CPU trials and
their independent audit. A fresh real GPU FP32 audit precedes sequential B1/B2
factorial comparisons: selected and combined frontends apply equally to Garnet
and vLLM-Metal, with all complete normalized outputs and canonical pixel hashes
required to match. Guards sample every 0.5 s and require a quiet interval.
Initialization, frontend setup and first request after initialization remain
separate from warm means.

## Initial actual inference results

PID 43673 terminated exit 0 after 468.5986 seconds. The independent audit
verified 360 archived files, one actual FP32 GPU audit and 48 complete matching
requests. Each runtime/frontend/batch has four warm requests from two fresh
workers, excluding each worker's first request. Decode is tokens/s per row;
B2 aggregate throughput is twice the per-row rate.

| Batch / frontend | Garnet TTFT ms | vLLM TTFT ms | Garnet decode | vLLM decode | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 selected | 202.1690 | 266.7926 | 48.3124 | 26.0961 | 1.85132 | 0.75778 |
| 1 combined | 190.5578 | 263.9491 | 48.4299 | 25.8111 | 1.87632 | 0.72195 |
| 2 selected | 359.7302 | 433.5916 | 44.6683 | 17.1350 | 2.60686 | 0.82965 |
| 2 combined | 344.6228 | 428.1885 | 44.6230 | 17.0136 | 2.62279 | 0.80484 |

Native warm TTFT is 5.74%/4.20% lower at B1/B2. Its preparation falls
45.1086→27.0568 ms and 49.7311→29.3910 ms. Comparator TTFT is 1.07%/1.25%
lower, with preparation 53.4348→29.4551 ms and 48.7116→29.1822 ms. Native
decode differs by +0.24%/−0.10%; no decode benefit is established by this CPU
optimization. B1 still misses 2x decode and both batches miss half TTFT. These
VL-only results do not replace the full three-model comparison.

Native initialization is 4.1053→4.1040 s at B1 and 4.1368→4.1296 s at B2;
first-request TTFT AFTER initialization is 353.2529→307.7481 ms and
566.6305→531.4282 ms. Comparator initialization is 33.3384→33.1618 s and
33.2325→33.3380 s; first-request TTFT is 307.7654→243.4624 ms and
527.8547→477.9529 ms. Frontend setup is measured separately, about 2.6–4.0 ms,
including module import for both policies in this harness. Lazy thread startup
is inside the first request. These are not hardware-cold observations.

Inference archive SHA-256:
`562ca72380d0f67d8cd4fb45ba602652defd42218d50da86588a2cc917402fe2`.
The independent audit was uploaded before confirmation PID 52244 started at
`logs/threaded-striped-inference-confirm-20261015`. The intervening GPU component
job had already terminated exit 0. Confirmation repeats the actual dtype audit
and both factorial comparisons; the selected frontend remains unchanged until
that evidence is audited.

## Completed confirmation

PID 52244 terminated exit 0 after 467.5469 seconds. The combined audit verifies
694 archived files, two actual GPU FP32 audits and 96 complete matching requests.

| Batch / frontend | Garnet TTFT ms | vLLM TTFT ms | Garnet decode | vLLM decode | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 selected | 204.4505 | 273.0357 | 48.2060 | 26.0424 | 1.85106 | 0.74881 |
| 1 combined | 186.7873 | 263.7694 | 48.3070 | 26.3112 | 1.83598 | 0.70815 |
| 2 selected | 365.0355 | 433.5095 | 44.6701 | 17.2996 | 2.58214 | 0.84205 |
| 2 combined | 347.3989 | 425.9809 | 44.6947 | 16.9642 | 2.63465 | 0.81553 |

Native TTFT reductions repeat: 5.74%/8.64% at B1 and 4.20%/4.83% at B2 in
initial/confirmation. Native decode is essentially flat (+0.24%/+0.21% at B1,
−0.10%/+0.06% at B2). Comparator warm TTFT is lower in both runs too. No
decode gain is established by these CPU changes. The full B1 decode and both
half-TTFT targets remain unmet, even though preparation itself is much faster.

Confirmation first-request TTFT AFTER initialization is native
355.6996→303.6347 ms / 561.7997→515.7798 ms at B1/B2, and comparator
305.4192→245.7952 ms / 511.1668→478.5231 ms. Initialization, frontend setup and
all first-request observations remain separate in the verified JSON. No
hardware-cold or shared-memory RSS attribution is claimed.

The new parallel-vision native GPU study explicitly applies this combined
frontend equally to control and candidate. General defaults and unsupported
image paths remain unchanged. Later native and comparator measurements must
use the same common frontend; this VL-only study cannot replace or be mixed
with the older full-three-model comparison.

Combined inference archive SHA-256:
`e3b095dd07e2ef417189ac50d91369e0cd75330daa7a8449b0a5c0ec2f93d4b1`.
Both terminal trials were archived before installing the subsequent optional
vision source changes; their actual worker source identities remain preserved.

The additional matched-FP32 comparison preserves learned values and FP32
arithmetic/activation/KV precision, while native compact weight storage differs
from comparator FP32 storage. It does not establish stock BF16 speedups. The
full three-model 2x decode / half-TTFT goal remains unmet and active.

```text
python3 queue_threaded_striped_probe.py ROOT ROOT/logs/threaded-striped-probe-confirm-20261015
python3 archive_threaded_striped_probe.py ROOT ARCHIVE
python -X utf8 verify_threaded_striped_probe.py CPU_RESULTS CPU_VERIFICATION_JSON
python3 launch_threaded_striped_inference.py ROOT ROOT/logs/threaded-striped-inference-20261015
python3 archive_threaded_striped_inference.py ROOT ARCHIVE
python -X utf8 verify_threaded_striped_inference.py INFERENCE_RESULTS VERIFICATION_JSON
```
