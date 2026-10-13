# Verified direct WorldSense launcher in the benchmark guard

After the packed-layout experiment, a release-test xlang3 process no longer
has a live parent containing `WorldSense/Backend/package_runtime.py`. The
existing guard correctly rejects that unverified process. Read-only process
inspection identifies an inline `-B -u -c` wrapper that runs the WorldSense
HTTP launch script directly. Raw command arguments and environment are not
printed or saved, and inline process code is never executed by the checks.

The comparison guard now supports this specific inspected mode. It requires
the expected release-test executable, exact inline-code SHA256
`751439d940f63c48dafa52f3a707c43c1584af6df675247dc0a5ae913dd6e42a`,
and exact entry-script SHA256
`9ac0a1fbdfefcba3349ecd5b0e5a00e63222f1cb588290ed55046d4851b70233`.
The script contains no configured inference-runtime markers. Both this mode
and the original ancestor mode still require a successful loaded-library
query and reject inference/GPU libraries; CoreML, ANE and BNNS are added to
that rejection list. Names or a `WorldSense` substring alone are insufficient.
The audit concerns the inspected launcher and currently loaded libraries;
it is not a claim about every possible request an HTTP application could serve.

Eighteen regression/read-only checks pass on the Mac, including altered
inline code, a name alone, the wrong executable, a missing PID, unreadable
libraries, ten inference/accelerator library families, changed entry source
and actual verification of the live inspected service. The initial checker
failed because its extracted-function test namespace omitted `json`; that
failed command/log/source is retained alongside the corrected passing run.
The production function already imports `json` normally.

This changes benchmark eligibility checks, not model arithmetic or timing.
No unrelated process is stopped or modified. The current service can be
verified, while unknown programs, changed fingerprints and active inference
libraries retain conservative rejection. The subsequent FP32 comparison
driver uses the same checked function. The full speed/TTFT goal remains unmet.
