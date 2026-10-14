# Threaded fresh WebP preparation: exact inputs, initial latency benefit

Two CPU scans preserve all model input bytes and reduce fresh preparation by
about 9 ms at both batches. A first matched native/vLLM run preserves 48
complete output sequences; native warm TTFT falls 2.6%/1.8% at batches 1/2.
The confirmation inference run is still live. No decode improvement or full
2x/half-TTFT achievement is claimed, and the default frontend is unchanged.

The optional CPU bridge uses Pillow's own bundled libwebp 1.6.0 with threaded
decoding requested, RGB output and all normal filtering/upsampling options.
It accepts only a conservatively parsed static WebP container with no metadata,
animation or alpha. All other images use the unchanged selected Pillow path.
Every request rereads and decodes the source, resizes with the same Pillow
bicubic operation, normalizes/arranges identical FP32 patches and writes/hashes
fresh patch/RGB files. No image/encoder/prefix reuse or reduced pixels/tokens
is introduced. Threading is CPU preparation, not concurrent model inference.

## Actual CPU results and exactness

| Batch | Initial selected → threaded ms | Repeat selected → threaded ms |
| --- | ---: | ---: |
| 1 | 44.45 → 35.40 | 44.40 → 35.45 |
| 2 | 46.66 → 37.54 | 46.48 → 37.64 |

The single-threaded native RGB alternative saves only about 2.3–2.6 ms, so
threading contributes additional work reduction. Two ABBA CPU scans retain
768 timed records. Each sample includes fresh source read/decode, exact resize,
normalization, patch layout, file writes and hashes. Both threaded and native
single-threaded routes match every pixel of the actual source RGB image and
its canonical patch SHA256. These are preparation times, not model TTFT.

The standalone compatibility suite passes 956 comparisons over 239 fixtures:
RGB/RGBA/L, lossy/lossless WebP, four sizes, orientations 1–8 plus absent EXIF,
animation, PNG/JPEG/BMP fallback and PNG EXIF after IDAT. Sixteen fixtures use
the native route; full decoded RGB also matches there. All resized RGB and
patch bytes match. Alpha/animation/metadata fixtures conservatively fall back.
Control/candidate open independently, so control loading cannot hide lazy
metadata behavior.

The first standalone test failed before image checks: the original bridge
retained Pillow's wheel build-time `/DLC/PIL/.dylibs/libwebp.7.dylib` ID. It
worked in the initial scan because Pillow already loaded the actual library.
The rebuilt bridge explicitly points to the existing bundled library using
`install_name_tool`; only the new experimental bridge is changed, with no
global loader environment or installed-package modification. Both original
source/logs and the fix are retained. The standalone retry and repeated CPU
scan pass. Corrected bridge SHA256:
`3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3`.

The [libwebp advanced API](https://developers.google.com/speed/webp/docs/api)
documents the threading option. [Pillow 12.3's decoder source](https://github.com/python-pillow/Pillow/blob/12.3.0/src/_webp.c)
uses default animation-decoder options; actual equality and timings above
are verified on the M4 rather than inferred from API availability.

## Initial actual matched inference

Fresh FP32 GPU audit, same learned values/prompts/image bytes/caps, greedy
stopping, fresh vision encoding, prefix/speculation disabled. Each runtime/
frontend has two fresh workers with three requests each, four warm samples
after excluding first use. Selected and threaded preparation apply equally
to Garnet and vLLM-Metal. Native uses the retained attention candidate at B1
and retained control at B2, identically across frontend policies.

| Batch / frontend | Garnet TTFT ms | vLLM TTFT ms | Garnet TPS/row | vLLM TPS/row | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 selected | 208.20 | 262.46 | 48.25 | 26.30 | 1.834 | 0.793 |
| 1 threaded | 202.75 | 251.15 | 48.29 | 26.48 | 1.824 | 0.807 |
| 2 selected | 385.95 | 442.22 | 44.75 | 17.16 | 2.608 | 0.873 |
| 2 threaded | 378.81 | 433.75 | 44.72 | 17.07 | 2.619 | 0.873 |

Native TTFT improves less than the CPU component saving; execution/protocol
variation also affects the observed endpoint. Decode is essentially unchanged.
The relative TTFT fraction does not improve: this helps both runtimes and
does not establish a larger advantage over vLLM. B1 decode remains below 2x,
and neither batch reaches half TTFT. B2 aggregate TPS is twice the per-row rate
for these equal rows; terminal predictions count and the first does not.

This remains an additional matched-FP32 study: native compact BF16 weight
storage versus losslessly expanded FP32 comparator storage retains different
weight widths/traffic. It is not a stock BF16 win. The actual audit confirms
FP32 parameters/activations/KV on the GPU. Complete prompt/output IDs, stopping
flags, fresh encoder identities and patch/RGB hashes match across 48 outputs.
One B2 attempt was interrupted by the inference guard (PID 61852 later absent)
and discarded before a clean retry. No unrelated job was stopped or guard
relaxed. Interrupted timings are excluded, with raw logs retained.

Worker initialization, first requests and CPU frontend setup are separately
recorded; frontend setup includes optional bridge loading before requests.
OS/driver caches were not flushed. No sums are presented as an observed
hardware-cold end-to-end latency. Broader images/tasks, longer contexts and
peak-memory savings are not established by these results.

## Evidence and live confirmation

Owned root: `/Users/zsdx/garnet-metal-investigation`. CPU scans and standalone
checks are terminal. Initial inference PID 57190 is terminal exit 0.
Confirmation PID 66500 was launched behind that successful predecessor;
inspect its actual process and `logs/webp-thread-inference-confirm-20261013/exit.json`
before continuing. Do not restart because observation times out. Its runner
is `run_webp_thread_inference.py ROOT NEW_OUTPUT_DIR`; exact commands are logged.

[CPU evidence](cpu/evidence/) verifies 53 file hashes, 768 timing records and
956 compatibility checks. CPU archive SHA256:
`45d50cb76ae3893cf92c1111dd75ed0d261b7af14d702a3a8cea170e601c0803`.
[Initial inference evidence](inference/evidence/) verifies 430 file hashes,
the same CPU proof, one actual dtype audit and 48 matched outputs. Archive:
`ba37c688d7547923d3903803935e1f560542dc4068fd0a022bb6f25bb77b0461`.
Raw arrays/bridge binaries are omitted with size/SHA proofs; source, compile/
loader commands, library/model identities, failed attempts, timings, versions,
monitors and exact outputs are retained. Existing helper-template scope labels
still mention copy elision; executed source and frontend policy identify the
threaded experiment.

```text
python test2026/metal/verify_webp_thread.py results/20261013/webp-thread/cpu
python test2026/metal/verify_webp_thread.py results/20261013/webp-thread/inference
```

Only optional investigation helpers were added. Shared selected preprocessing,
native serving code/libraries, installed Pillow and protected GPT-OSS/Pi
checkouts remain untouched. The full three-model goal remains active and unmet.
