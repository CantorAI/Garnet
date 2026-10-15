# Original-order vision context prefetch

Two actual sequential component trials admit the eight-key prefetch kernel
with bit-identical FP32 output. Loads are prefetched while the FMA accumulator
retains the original key order. Softmax is unchanged. The separate Metal library
disables fast math. This avoids the changed reduction order that failed the
earlier trained-grid parallel-context experiment.

The first trial measures five variants over 24 shapes; the independent repeat
measures the selected eight-key variant over the same 24 shapes. All 144
shape/variant checks pass exact output-bit equality and the original FP64
reference bound across four input generations. There are 23,040 timed
submissions, with raw GPU and wall observations preserved in evidence.

For the serving shape (216 tokens, 12 heads, head dimension 64), combined
softmax/context GPU time falls 19.7–34.7% at B1 and 19.4–26.3% at B2 across
the two trials. Wall time falls 14.9–27.1% and 16.7–22.7%, respectively.
These are component measurements, not whole-model speed or TTFT gains.

An optional native integration is OFF by default and admits only this trained
shape at batch 1 or 2; other shapes retain the original route. The actual
trained full-grid gate subsequently passed bit-identical embedding values at
both batches, including two fresh encodes and small-grid fallback/close counts.
Full original-model validation and six-profile native ABBA are still pending
at the time of this record. No new vLLM comparison or serving gain is claimed.

Actual component jobs: PID 62873 exited 0 in 58.6223 s; PID 64132 exited 0 in
18.3778 s. The combined archive SHA-256 is
`3348dc7c205ce8db5c946352d747ef03e61f9eb1155772d60fd37f3e33108351`.
The independent audit checks 51 archived files. Commands and exact generated
shader/harness sources are retained with the raw observations.

```text
python3 launch_serial_vision_context.py ROOT ROOT/logs/serial-vision-context-20261015
python3 launch_serial_vision_context_confirm.py ROOT ROOT/logs/serial-vision-context-confirm-20261015
python3 archive_serial_vision_context.py ROOT ARCHIVE
python -X utf8 verify_serial_vision_context.py EVIDENCE VERIFICATION_JSON
```
