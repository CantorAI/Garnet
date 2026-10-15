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

## Trained component follow-up prepared

`make_bf16_feed_mlp_component.py` keeps the prior complete dependent MLP
harness and its existing FP32 stage-rounded CPU reference, with unchanged
native/reference tolerance coefficients and all four input patterns. Weight
placeholders/constants use original BF16 bits and explicit FP32 casts. Before
any timing, a separate GPU cast roundtrip must reproduce every original FP32
learned-value bit for the12 MLP tensors in the four fixture blocks: text/coder
layer0 and VL layers0/3. This is complete per-fixture tensor coverage, not
coverage of all model layers. Activations and arithmetic retain FP32 assertions.

Expected coverage is48 aggregate numerical records and12 first graph calls;
only profiles passing every original native/reference/candidate predicate may
enter two ABBA timing rounds. Prior stress/reference failures remain recorded;
no failing profile may be removed or its coefficients widened. Wall clocks
include one synchronous graph call per MLP, with setup/first costs separate.
These component times cannot establish model TTFT, memory savings or a vLLM
ratio, even if a storage policy improves them.

Local source generation and expanded driver/auditor syntax checks passed.
No Mac compilation or execution is claimed. Exact queue PID60730 observes
controller53701 and requires its audited completion before launch, then waits
for the actual returned component handle and freezes/audits the outcome.

```text
python3 queue_bf16_feed_mlp_component.py ROOT ROOT/logs/queue-bf16-feed-mlp-component-20261015
# Only after the primitive's terminal archive and audit:
python3 launch_bf16_feed_mlp_component.py ROOT ROOT/logs/bf16-feed-mlp-component-20261015 component
python3 archive_bf16_feed_mlp_component.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_bf16_feed_mlp_component.py AUDIT AUDIT_RESULT
```
