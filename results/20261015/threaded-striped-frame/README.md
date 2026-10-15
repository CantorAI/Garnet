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
separate from warm means. This study is not yet an inference success claim.

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
