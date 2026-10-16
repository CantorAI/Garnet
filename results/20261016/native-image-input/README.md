# Native image preparation, measured through XLang3

The external CPython/NumPy/Pillow image subprocess has been removed from Garnet Metal serving. The replacement is an in-process C++ JPEG/PNG operator exposed as `garnet.prepare_image`, called from the existing XLang3 model-serving script. Learned vision/decoder graphs still execute on native Metal/MPS. This change does not replace XLang3 with another inference engine or claim that all source files cease to use Python syntax.

## Implementation and input contract

Native decoding uses libjpeg-turbo3.2.0 and libpng1.6.59, with EXIF orientation, separable fixed-point bicubic resizing, FP32 normalization and merge/temporal patch layout. It returns a host FP32 tensor directly to the native vision graph, removing the per-frame helper process and patch-file round trip. JPEG warnings indicating damaged input fail closed. Input lengths, dimensions, tensor size, profile and normalization are bounded/validated. Source/error recovery is per invocation.

The initial format contract is JPEG and PNG, including the tested grayscale, CMYK JPEG, alpha/palette PNG,16-bit grayscale PNG and trailing-EXIF cases. Other formats are rejected without a Python fallback; broader formats are follow-up work. JPEG/PNG are native build dependencies, discovered only for the Metal xModel target. The source has a standard-library-only unsupported stub outside Apple Metal, so the CUDA source glob does not acquire JPEG/PNG dependencies. A no-Metal compile/symbol check was executed; a Windows CUDA build was not executed on this Mac.

Native preparation uses strict float compilation without contraction. The runtime's model-specific normalization/geometry comes from official model configs. The older Pillow/NumPy script remains only as an offline reference. The production runtime does not read GARNET_METAL_FRONTEND_PYTHON; that variable was removed from the local Cantor launch agent.

## Quick metrics

Tests are executed by the native XLang3 executable. Offline golden generation uses Pillow/NumPy; it is not a new production dependency. Alternating old-helper/native calls measure five samples per image and include the real old process startup/config/import/file path. Native samples include tensor creation. These component measurements are not VLM inference latency.

| Fixture | Old helper median | Native median |
|---|---:|---:|
| bunny | 284.67ms | 3.69ms |
| car | 275.12ms | 2.12ms |
| traffic | 282.58ms | 5.21ms |

Component wall time is about54–130times lower in these cases. That factor must not be applied to complete inference.

Two fresh control/candidate cohorts use the previously deployed seven-kernel/shared-vision profile as control. Runs are sequential control/candidate/candidate/control, with WorldSense/Cantor paused. Each fresh worker runs bunny, car, traffic, bunny; first-after-init is retained separately, and warm medians use the last three calls. Same model/checkpoint, prompt,1536+256 capacity,180-token cap,65536 image pixels and16-token prefill. Every frame is freshly decoded/encoded. No hardware cache flush or cached answer is claimed.

The predeclared native-port gate requires exact patch/output parity, no external helper in the candidate, and no median total/decode regression beyond5% in either cohort. This architecture-port preservation gate differs from the prior pure-kernel optimization screens and was recorded before timing. It is not proof of a general throughput win.

| Cohort | Median TTFT control → native | Median total control → native | Decode tokens/s control → native |
|---|---|---|---|
| 1 | 2.369 → 2.108s | 9.450 → 9.457s | 11.550 → 10.989 |
| 2 | 2.307 → 2.093s | 9.358 → 9.201s | 11.564 → 11.357 |

First-token medians improve9–11%; total medians are effectively flat (0.07% slower and1.67% faster). Decode medians are4.86% and1.79% lower, within the declared port-preservation limit, so no decoder-throughput improvement is claimed. First requests include setup variance and are not uniformly faster. Initialization, all individual calls and separate maximum-RSS/process-footprint values are preserved.

## Correctness and integration

Three real video frames have1,173,504 bit-identical FP32 patch values versus the saved reference. Twenty-one synthetic format/orientation/geometry fixtures also match exact patch bits, including EXIF after PNG IDAT. Corrupt/truncated JPEGs and invalid normalization are rejected. The final native test runner is test2026/metal/test_native_image.py, invoked by XLang3 against offline reference manifests.

All eight complete generated outputs per policy match token-for-token in the two timing cohorts. A separate request matches all state parts across32 decoder layers bit-for-bit, and three1.7B text prompts match complete tokens. These are sampled tests, not exhaustive numerical equivalence or semantic accuracy coverage.

The final main-source build additionally passes native matmul and all21 parity fixtures. Serving-API checks accept both a file path and encoded bytes, with the helper environment unset, and produce matching model text with image_preprocess_backend=native-cpp. After deployment, WorldSense has four online file sources and a fresh stored4B observation reports native-cpp. The first live call is a setup/cold sample (~22.9s), not a matched performance comparison. Camera remains unmounted. Existing portal/default tenant remain configured.

The old library/source and launch agent are backed up locally for rollback. The final build is out/build/garnet-metal-native-image. Future staging should rebuild Garnet with the native codec dependencies. The parent build cache retains the same seven optimization flags; chunk32 was not promoted.

## Previous rejected chunk32 attempt

Before the native-port work, extended-backend chunk32 passed full output comparisons and reduced TTFT, but improved total only1–3% and failed a decode-preservation gate in one cohort. It was not deployed. Its separate work/4b-chunk32 selection, clocks and failure decision are retained; no32-layer state pass is claimed because that attempt's validation driver selected the wrong source admission path and stopped. The production profile remains chunk16.

## Limits

This is a local Mac build and small fixture suite. Broader format/color-profile/interlaced/corrupt-image coverage and packaging across hosts remain necessary. Homebrew codec libraries must be available at runtime. No vLLM comparison, general speed ratio, production semantic accuracy guarantee or full native Windows test is claimed. The legacy global research verifier still has its previously documented historical entry-point hash assertion; it was not weakened or reported as passing.
