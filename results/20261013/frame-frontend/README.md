# Fresh-frame copy elision: less CPU work, no clear model latency gain

The common frontend loads image data before reading orientation, applies
EXIF transforms only for orientations 2–8, and converts only non-RGB images.
Plain RGB frames avoid the two whole-image copies made by the previous
`ImageOps.exif_transpose(...).convert('RGB')` chain. Pillow bicubic resizing,
normalization, temporal/channel/merge ordering, little-endian FP32 patches,
input image bytes, resolution and token limits are unchanged.

Every request still opens and decodes its source afresh, prepares new patch/
RGB files, and runs a fresh vision encoding. This is not image/encoder/prefix
reuse. The default common frontend applies the same implementation to both
Garnet and vLLM-Metal. `frame_preprocessing_control.py` preserves the original
source byte-for-byte for A/B reproduction; an optional frontend argument to
the observed frame protocol selects `control` versus `selected`.

## Attribution and alternatives

The frozen `RealWorld-04.png` asset is actually WebP, determined from file
content and Pillow's format. Initial commentary inferred PNG from its name;
that inference was corrected. The generic attribution labels in raw data
remain decode/EXIF/convert, not a claim about a PNG decoder.
Warm CPU attribution over eleven non-first samples per batch reports:
decode/EXIF/convert 33.5 ms, resizing 11.3 ms, normalization roughly 1.2 ms,
patch layout about 0.9 ms, with total fresh preparation/write/hash roughly
53.0/56.7 ms at batches 1/2. Arrays are C-contiguous; noncontiguous element
writes were not the cause.

Two CPU ABBA scans check 384 timed records each. Borrowing RGB input instead
of making the copies reduces preparation from 49.69 -> 45.40 ms and
51.76 -> 49.72 ms initially, then 48.31 -> 45.09 ms and
51.56 -> 48.96 ms. The direction repeats; this is preprocessing, not model
TTFT or decode speed.

OpenCV 5.0.0 was already installed in the isolated environment; no packages
were changed. The initial conservative PNG-only candidate fell back on this
WebP asset, so that run does not measure OpenCV decoding. The corrected
simple-WebP candidate matches exact pixels/patches but improves little over
copy elision (44.77/47.07 ms against 48.64/50.67 ms control). It is not chosen
for the shared frontend: extra decoder selection/metadata handling is not
justified by that marginal extra signal. Production preprocessing imports
only its existing Pillow/NumPy dependencies.

The initial scan's `opencl_used` output label actually recorded configuration
state, not GPU execution; its fast decoder branch never ran. The corrected
scan explicitly disables OpenCL and names that field
`opencl_enabled_after_run`. CPU image decoding is not learned inference.

## Exactness and actual matched inference

Three compatibility passes cover 448 cases each: RGB, RGBA, L, P, I and CMYK,
four sizes, all eight orientations and no-orientation cases, plus real input
at batches 1/2. Seven PNGs place EXIF after IDAT. The later tests open control
and candidate separately for those lazy files, so control loading cannot
hide an orientation-read bug. The integrated implementation is checked
against the preserved original function. Fourteen additional conservative
CV fallback comparisons per pass also match. All RGB and patch bytes match
exactly; the frozen patch SHA256 is unchanged.

A sequential factorial ABBA run alternates frontend policy within each
runtime, at batches 1/2, using the same M4 / 16 GiB and Qwen3.5-0.8B values.
There are eight fresh persistent workers per batch, three requests each,
and four non-first samples per runtime/policy. All 48 complete normalized
token sequences, prompt IDs, finish flags and patch/RGB hashes match. Native
prefill plans remain 80 at B1 and 32+32+16 at B2. Prefix/speculative caches
are disabled; all comparator encoder identities are unique across rows and
requests within each worker. No competing inference runs simultaneously.

| Policy / batch | Garnet TTFT ms | vLLM TTFT ms | Garnet steps/s/row | vLLM steps/s/row | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original / 1 | 218.58 | 264.70 | 48.44 | 26.07 | 1.858 | 0.826 |
| Selected / 1 | 218.93 | 266.25 | 48.58 | 26.45 | 1.837 | 0.822 |
| Original / 2 | 389.63 | 443.82 | 44.75 | 17.16 | 2.608 | 0.878 |
| Selected / 2 | 387.75 | 442.68 | 44.75 | 17.12 | 2.614 | 0.876 |

All phase/worker records are in the evidence.
Selected/original TTFT fractions are 1.00158/1.00585 for Garnet/vLLM at B1,
and 0.99518/0.99743 at B2. There is no useful end-to-end gain established by
this four-sample study, despite reduced preprocessing work. Native B1
preparation falls 49.57 -> 46.67 ms; comparator 53.43 -> 49.50 ms. Other
execution variation offsets this small saving in observed first-token time.
No model gain is assigned to changing a CPU copy that runs before decode.

This is an additional matched-FP32 study, not a stock BF16 win. Native
compact BF16 storage and FP32 arithmetic versus exact F32-expanded comparator
weights retain their known storage-width caveat. Actual equivalence proofs,
versions, configured dtype and complete outputs remain. B2 aggregate
throughput is twice the per-row prediction rate; terminal prediction counts
and first prediction/prefill does not. B1 decode remains below 2x and neither
batch reaches half TTFT. The full three-model objective remains active.

The simpler copy-elision code is retained because its exact-input CPU work
reduction repeats without an added decoder dependency. GPU serving library
and learned operators are unchanged. Startup and first-use are separate in
records; OS/driver caches were not flushed. Peak child RSS is not aggregate
or unique GPU memory. No larger-model, long-context or general video-quality
claim follows from these fixed image/workload results.

## Commands and provenance

```text
python3 profile_frame_frontend.py INVESTIGATION_ROOT OUTPUT_DIR
python3 compare_frame_frontends.py INVESTIGATION_ROOT OUTPUT_DIR
python3 test_frame_frontend_exact.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_frame_frontend_inference.py INVESTIGATION_ROOT OUTPUT_DIR
python3 archive_frame_frontend.py INVESTIGATION_ROOT ARCHIVE_PATH
python3 verify_frame_frontend_evidence.py EVIDENCE_DIR PROOF_JSON
```

The first attribution job failed before timings due to a wrong case-file
path; the corrected retry exits zero. All subsequent jobs exit zero. Exact
commands, identities, equivalence proofs, versions, numerical/timing/output
records and monitors are preserved. `source-history/` retains the failed
path version, PNG-only/initial WebP candidates and pre-integration sources;
current helper imports preserve the original control for reproduction.
Raw image/patch/matrix assets are omitted with hashes/sizes retained.
Only owned workers can be stopped on error; 0.5-second monitoring cannot
exclude arbitrarily brief external tasks. Protected GPT-OSS/Pi checkouts and
the original Mac checkout are untouched.

Unchanged selected library SHA256:
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
Archive SHA256:
`6908f3a0c7b4be82cf47c94a0e60235d6a3dcc39e93a3f303b2babb59009dab8`.
