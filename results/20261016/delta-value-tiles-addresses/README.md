# Wider delta value tiles CPU prerequisite

Status: local CPU addressing and literal-expression checks pass. No GPU compile,
numerical-equivalence result or performance measurement is established.

Hypothesis: widen the existing gated-delta recurrence's value tile from 16 to
32 or 64 columns per 128-thread group. Reuse the original Q/K normalization
prefix across two or four bands while retaining the original per-value float4
recurrence. This could reduce duplicated normalization and barriers, at the
cost of more live registers and potentially lower occupancy. The accepted
kernel already uses a 16-column tile; this is not scalar-to-vector conversion
or the previously rejected extra-dispatch normalization cache.

Generate all three variants from the accepted d52 source. CPU enumeration
reads each generated tile/column expression, checks unique ownership of every
state and output word, aligned float4 accesses and padding guards, and recovers
each literal original arithmetic band. It covers 12 profiles: widths 16/32/64,
batches 1/2 and padding 0/4 floats, with 16 heads and key/value dimensions 128.
Output indexing covers chunks 1/16/80 in each profile.

The first audit exited 1 before enumeration: its membership assertion wrongly
required renamed candidate variables in the unchanged width-16 control.
Freeze that [failure](failure/evidence/failure.json), then fix only the assertion
to require the original block for width 16. The generator and generated shader
remain byte-identical between the failed and corrected CPU runs.

The corrected audit and its independent frozen-source reproduction pass:
4,718,592 state words and 3,575,808 output words. Its 2.047-second duration is
CPU audit time, not kernel latency. The report's `original_kernel_sha256`
identifies the baseline with its entry renamed to `gated_delta_chunk_value16`.
[Frozen report](verified/evidence/report.json),
[independent reproduction](verified/verification.json),
[source and recurrence provenance](source-provenance.json).

The extracted original recurrence is byte-identical in accepted d52 and the
private paired-norm/RoPE candidate. All prototype tools and this audit stayed
on Windows while the Mac native-validation/controller tools were frozen.

Archive identities:

| Evidence | SHA256 |
| --- | --- |
| Original CPU assertion failure, 5 files | `c2334e138c7413b49975cf12d119fa4ed852ee1c41a2a0ffb34f6ce62815c5ef` |
| Corrected CPU prerequisite, 7 files | `df64029c07bd8e2128969645e27a8377f556f27e3cf322738fcba65a1d8aaffa` |

```text
python -X utf8 test2026/metal/audit_delta_value_tile_addresses.py src/backends/metal/metal_context.mm .work/delta-value-tile-addresses-fixed-20261016
python -X utf8 test2026/metal/archive_delta_value_tile_addresses.py .work/delta-value-tile-addresses-fixed-20261016 .work/delta-value-tile-addresses-fixed-20261016.tar.gz
python -X utf8 test2026/metal/verify_delta_value_tile_addresses.py results/20261016/delta-value-tiles-addresses/verified results/20261016/delta-value-tiles-addresses/verified/verification.json
```

Next gates are actual GPU compilation and original-output/state bit checks,
learned carried-state references, calibrated timing with stable own-nulls,
unchanged confirmation and complete native/model validation. Wider registers
may make this slower. This prerequisite changes no serving source or target
count; the full matched-model goal remains unmet.
