# BF16 graph feeds with exact FP32 casts: prepared, execution gated

Hypothesis: feeding original BF16 buffers to MPSGraph may avoid the retained
expanded FP32 weight cache used by the rejected command-buffer MLP trial.
The graph explicitly casts every weight tensor to FP32 before multiplication;
activation, product, nonlinear and result tensors retain FP32 dtype assertions.
Original learned values are unchanged. This is a storage/dataflow experiment,
not a lower-precision arithmetic mode. Internal compiler allocation, actual
memory savings and speed are unknown until measured.

Installed SDK inspection found `MPSDataTypeBFloat16` in MPSCoreTypes.h,
available from macOS14. The host runs macOS15.6.1. An initially guessed
MPSTypes.h path did not exist; enumerating the installed framework located
the actual Versions/A/Frameworks/MPSCore.framework/Versions/A/Headers path.
This header discovery is not runtime support proof. The new probe also
archives the installed header alongside the generated source.

`make_mlp_bf16_feed_graph.py` derives from the already validated private
MPSCommandBuffer implementation. It removes the explicit expanded FP32 weight
allocation, keeps original weight owners alive, feeds their tightly packed
BF16 buffers, and inserts exact FP32 casts. Native producers, refreshed MPS
command-buffer roots, final completion/error checks and abort barriers stay
unchanged. The macOS availability guard becomes14 for the new dtype.

The primitive retains all64 original safety/ownership cases and64 dependent
two-graph cases with native producers/consumers. Original native comparison
bound5e-6+5e-5*abs(native), supported/fallback rows, pending reads, foreign
writers, alias rejection, uninitialized input, counter reset and abort recovery
remain mandatory. The auditor regenerates every private source from its
archived original and checks both storage and arithmetic dtype paths.

Local transformation and expanded-driver/auditor syntax checks passed. No Mac
compile, runtime execution, trained-model numerical pass, memory reduction or
speed gain is claimed. The earlier [cast-half projection experiment](../mpsgraph-prefill/README.md)
was slower; this BF16-storage dependent graph is a distinct test and must earn
its own evidence. The [FP32-cache command-buffer integration](../mlp-command-buffer-native/README.md)
stays rejected regardless of this probe's future result.

Controller PID53701 at ALT/logs/shape-native-bf16-finalize-20261015 waits for
exact controller47970, then the exact selected-shape native worker. It archives
and audits all72 complete native outputs before admitting this probe, enforces
its own quiet interval, waits for the primitive's exact returned handle and
preserves success/failure before audit. Neither this queued dependency nor
controller liveness establishes that the primitive executed.

```text
python3 finalize_shape_native_and_probe.py ALT ALT/logs/shape-native-bf16-finalize-20261015
# Only after selected-shape native terminal archive and full audit:
python3 launch_mlp_bf16_feed_primitive.py ROOT ROOT/logs/mlp-bf16-feed-primitive-20261015 primitive
python3 archive_mlp_bf16_feed_primitive.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_mlp_bf16_feed_primitive.py AUDIT AUDIT_RESULT ARCHIVE
```

ROOT is /Users/zsdx/garnet-metal-investigation; ALT is its
shape-selected-two-investigation-20261015 child. Exact executed commands and
runtime metadata must supplement these templates. Current inference inputs
remain frozen, and competing runtimes/probes remain sequential. A primitive
pass would justify trained numerical/performance testing, not promotion or
a vLLM/model-speed claim. The full three-model2x/half-TTFT goal remains unmet.
