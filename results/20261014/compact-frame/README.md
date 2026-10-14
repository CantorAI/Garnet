# Optional compact frame preparation

On the actual M4, fresh image preparation decreased from 49.151 to 48.071 ms
at batch 1 and from 51.397 to 50.259 ms at batch 2 (about 1.1 ms, 2.2%).
These are CPU-only measurements, not image inference or a new vLLM win.
The selected common frontend remains unchanged pending inference evidence.

The candidate retains the same image load, EXIF transform, RGB conversion,
Pillow BICUBIC resize and geometry. NumPy `out=` avoids normalization arrays
while preserving the three separately rounded FP32 operations and order.
Temporal/batch broadcasting avoids redundant copies before the final contiguous
little-endian FP32 patch buffer. Learned operations and image resolution are
unchanged; every frame is freshly prepared.

Compatibility passed 768 combinations of RGB/RGBA/L, EXIF orientations 1–8,
four source sizes, B1/B2, temporal sizes 1/2 and two mean/std settings. Every
RGB/patch byte, shape, dtype and contiguity check passed. The actual frozen
image then passed 192 timed preparations with canonical image and patch hashes.
NumPy 2.3.5 and Pillow 12.3.0 ran in the isolated comparator environment.
Each batch used two ABBA rounds with twelve samples per window, after at
least 250 ms common warming. Timing includes fresh load/decode, preprocessing,
buffer file writes and hashes. OS caches were not flushed. Guards reject other
inference/probe jobs; this CPU trial ran after native inference was terminal.

Initial owned PID 33681 exited 1 before timing: the shared guard needed a
global `output` alias. Only `output=out` was added to the runner. Repaired
PID 37472 exited 0 in 50.38 seconds. The failure log is preserved alongside
the successful numerical and timing evidence. No approval/guard was weakened.

`verification.json` independently verifies all 22 archived files, compatibility
coverage, sample identities and timing means. Archive SHA-256:
`d968060d8d9bf4ccda246efb312d8bc4e32664b902769be109e23cc8b60a4afd`.
Commands and executed sources are preserved in `evidence/`.

```text
python3 queue_compact_frame_probe.py ROOT ROOT/logs/compact-frame-probe-20261014
ROOT/comparators/vllm/venv/bin/python ROOT/source/test2026/metal/run_compact_frame_probe.py ROOT ROOT/logs/compact-frame-repair-probe-20261014
python3 archive_compact_frame_probe.py ROOT ARCHIVE
python verify_compact_frame_probe.py RESULTS VERIFICATION_JSON
```
