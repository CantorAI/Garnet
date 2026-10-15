# Exact-coordinate parallel Pillow resize

Two actual CPU trials preserve complete resized RGB and FP32 patch bytes and
reduce fresh preparation by about 7.5–7.9 ms. Their independent audit covers
36 archived files, 1920 compatibility comparisons and 384 timed preparations of
the frozen image. The first actual image-plus-text study passed 48 complete
matching requests and shows lower native warm TTFT at both batches. A full
confirmation is running. The full goal remains unmet.

| Batch | Initial selected → strips ms | Repeat selected → strips ms |
| --- | ---: | ---: |
| 1 | 49.3286 → 41.8515 | 49.4151 → 41.9380 |
| 2 | 51.6332 → 43.9928 | 51.8046 → 43.9359 |

Each CPU sample includes fresh source opening/decoding, preprocessing, patch/RGB
file writes and hashes. Both trials have two ABBA rounds per batch and twelve
samples per window, after common warming. These are CPU preparation times, not
model TTFT or decode rates. OS caches were not flushed.

## Method and exactness limits

The optional module uses a persistent four-worker CPU thread pool and Pillow's
own bicubic resizer on vertical output strips. It admits only Pillow 12.3.0 RGB
images with bounded dimensions, at least 262144 source pixels, a changed height,
and a reduced vertical scale denominator that is a power of two. Output cuts
are multiples of the reduced denominator, so source-box boundaries are integers.
Other geometries/modes/versions retain the selected Pillow operation.

Pillow's [resize wrapper](https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/src/_imaging.c)
passes float32 boxes. Integer boundaries within the admission bounds are exact
there. Its [resampler](https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/src/libImaging/Resample.c)
computes centers in double precision. The admitted dyadic scale and bounded
integer coordinates preserve center arithmetic exactly. Horizontal resize
geometry remains unchanged, and Pillow keeps the source-row filter halos for
each strip. This restriction avoids arbitrary fractional boxes that could
change coefficients or pixels. It is not a claim about all Pillow versions.

The actual source is 1372×1818 pixels; resized output remains **192×288 pixels
(width×height)**. The vertical scale is 101/16. Four output strips span
64/80/64/80 rows, with integer source boundaries 0/404/909/1313/1818. The same
Pillow filter, dimensions, RGB bytes, normalization order and FP32 patch layout
remain. Source images are fully decoded before concurrent readers start, and
all readers are drained before the caller may close the source.

Each compatibility run covers five source sizes, RGB/RGBA/L, all eight EXIF
orientations, B1/B2, temporal sizes 1/2 and two mean/std settings. Independent
source opens prevent shared lazy loading from hiding mistakes. All 960 cases
match every RGB and FP32 patch byte; 96 actually take the strip route and 864
exercise fallback. Every timed real-image candidate takes the strip route and
matches the canonical patch hash. No image or encoder output is reused.

Actual owned CPU jobs:

| Trial | PID | Exit | Wall seconds |
| --- | ---: | ---: | ---: |
| striped-frame-probe-20261015 | 17019 | 0 | 58.9717 |
| striped-frame-probe-confirm-20261015 | 19417 | 0 | 58.9505 |

Initial archive SHA-256:
`74321746a280938baade54a9a261f99851c08f53308e3468ea71c3b194355848`.
Combined CPU archive SHA-256:
`3c3183f873854919801f1a872c04e1063e845373b69e63113920d651419e0e26`.
Raw timing arrays, compatibility records, executed sources and independent
verification are under `cpu/`.

## Actual inference study in progress

Owned PID 21826 at `logs/striped-frame-inference-20261015` performs a fresh actual
FP32 GPU dtype audit, then B1/B2 sequential factorial ABBA comparisons. Both
Garnet and vLLM-Metal receive selected or striped common preprocessing equally.
Garnet uses the same selected global-alpha dylib for both policies, SHA-256
`887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f`.
The optional body-shape route is disabled. Models, arithmetic precision,
prompts, pixels, output caps and stops remain frozen. Images are freshly
encoded and cross-request prefix/speculative reuse is disabled. Complete
outputs and unique comparator encoder identities must match before accepting
timings. Initialization, frontend setup and first request after initialization
are separate from warm means. Competing inference runtimes never intentionally
run simultaneously, and the existing guard and contention retries remain.

This remains an additional matched-FP32 study: learned values restore exactly,
while native compact BF16 weight storage and comparator expanded FP32 storage
have different widths and memory traffic. It is not a stock BF16 claim.
The selected common frontend and GPU sources remain unchanged pending evidence.

The initial inference job terminated exit 0 after 472.4574 seconds. Its actual
FP32 GPU audit and all 48 complete requests passed independent verification
over 353 archived files. No profile required a contention retry. Each runtime,
frontend and batch has four warm requests from two fresh workers; first requests
are excluded. Decode is tokens/s per row, with B2 aggregate twice that rate.

| Batch / frontend | Garnet TTFT ms | vLLM TTFT ms | Garnet decode | vLLM decode | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 selected | 205.1015 | 263.4634 | 48.2790 | 25.9573 | 1.85994 | 0.77848 |
| 1 strips | 199.7885 | 256.8373 | 48.3099 | 26.0940 | 1.85138 | 0.77788 |
| 2 selected | 359.3341 | 433.8292 | 44.7026 | 17.1509 | 2.60644 | 0.82828 |
| 2 strips | 355.1581 | 435.8267 | 44.6909 | 17.2465 | 2.59131 | 0.81491 |

Native warm TTFT is 2.59%/1.16% lower at B1/B2. Actual native preparation falls
45.4195→37.4046 and 48.7921→41.3298 ms. Comparator preparation also falls, but
its B2 endpoint TTFT increases 0.46%; other model/protocol variation offsets
the CPU saving there. No decode benefit is established by this CPU change.
Neither batch achieves half TTFT, and B1 decode remains below 2x. This VL-only
study does not replace the last full three-model comparison or permit combining
new native numbers with older comparator measurements.

Frontend setup is about 2 ms for strips and is reported separately from worker
initialization. Its lazy thread startup is included in the first actual request.
Native first-request TTFT after initialization is nearly flat at B1
(355.4569→355.3138 ms) and lower at B2 (569.4517→557.6288 ms). All initialization,
first-request and warm observations remain separate in `inference/verification.json`;
OS/driver caches are not flushed, so these are not hardware-cold observations.

Initial inference archive SHA-256:
`7b00d410349c8c5b699640402bda245c83e495b59e30139ea0af14b413add89a`.
The independent initial audit was uploaded before confirmation PID 30402 at
`logs/striped-frame-inference-confirm-20261015`. Confirmation repeats the actual
GPU dtype audit and both B1/B2 factorial comparisons with the same source,
selected GPU library, canonical image bytes, stops and caps.

```text
python3 launch_striped_frame_probe.py ROOT ROOT/logs/striped-frame-probe-20261015
python3 archive_striped_frame_probe.py ROOT ARCHIVE
python -X utf8 verify_striped_frame_probe.py CPU_RESULTS CPU_VERIFICATION_JSON
python3 launch_striped_frame_inference.py ROOT ROOT/logs/striped-frame-inference-20261015
python3 archive_striped_frame_inference.py ROOT ARCHIVE
python -X utf8 verify_striped_frame_inference.py INFERENCE_RESULTS VERIFICATION_JSON
```
