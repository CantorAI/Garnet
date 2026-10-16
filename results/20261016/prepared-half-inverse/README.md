# Exact half inverse CPU verification

The proposed integer inverse reconstructs original FP32 bits from a finite
half word and a scale without a floating conversion and ldexp. Exhaustive
CPU tests pass on Windows MSVC x64 and Apple clang ARM64, including the
Mac CLZ branch. The helper is not installed in a serving runtime and has no
measured setup-time or inference speed benefit yet.

Both hosts execute the same header, SHA256
`2ba0f45baff48f129743b7ee1c54c941ef8a0ec43e58ed686670a8a72882f7c5`,
against the unchanged original EncodeExactScaledHalf codec and an independent
floating reconstruction. Signed zero, finite normal and subnormal half words,
all scales from -64 through 64, nonfinite rejection and invalid-scale output
preservation are required.

| Exhaustive case class | Cases | Exact or admitted | Rejected |
| --- | ---: | ---: | ---: |
| Original BF16 codec across 129 scales | 8454144 | 1122816 | 7331328 |
| Half words across 129 scales | 8454144 | 8189952 | 264192 |
| Six invalid scales across all half words | 393216 | 0 | 393216 |

Every admitted original-codec result reconstructs raw BF16 bits shifted left
16 exactly. Every finite half result matches the independent floating inverse;
all rejections leave output unchanged. The invalid scales include INT_MIN,
-1000, -65, 65, 1000 and INT_MAX.

Windows MSVC uses /O2 and /fp:precise. Mac uses /O3, -fno-fast-math and
-ffp-contract=off on the actual M4 with 16 GiB. Invocation, compiler identity,
preprocessor macros, source snapshots, command logs and exhaustive reports are
captured. [Windows audit](windows/verification.json),
[Mac audit](macos/verification.json).

The first Mac compile exited 1 because invoking the resolved clang++ binary
directly did not locate cstdint in this SDK installation. Preserve that failed
attempt before changing the runner to /usr/bin/xcrun clang++. The retry uses
the identical mathematical header and succeeds. This is a toolchain fix;
neither the codec nor numerical acceptance changed.
[Failed compile archive identity](macos-compile-failure/archive-verification.json).

Frozen archive SHA256 values:

- Windows: `f038b2e04bde9c816fba9384543d9aba26df424c908cfdf05fdf4d221a14bc10`.
- Mac failed compile: `0815d96ce7e311d366bce35325166ba8683f0bda2cae2de316afb8ceea2e7a6b`.
- Mac successful retry: `f6adb209df42be096b375d053bc4d205f1247c2f2516544ec3c3c84cc327aa56`.

All three archives are extracted and verified on Windows; the successful Mac
archive is also independently audited on the Mac. These tests establish the
mathematical CPU prerequisite. Learned packing equivalence, setup latency and
any changed GPU or complete-model runtime still require separate evidence.
