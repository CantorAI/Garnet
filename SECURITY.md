# Security Policy

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's **Security** tab and select **Report a vulnerability** to send a
private report to the maintainers. Include affected versions or commits,
reproduction steps, impact, and any suggested mitigation. Please avoid sharing
credentials, private model data, or other unrelated sensitive information.

The maintainers will acknowledge the report, investigate it, and coordinate a
fix and disclosure when appropriate. No response-time guarantee is currently
provided.

## Scope

Security reports may cover Garnet source, xModel loading, model and engine
artifacts, native interfaces, backend integrations, memory safety, and unsafe
handling of untrusted inputs. Vulnerabilities in CUDA, TensorRT, OpenVINO,
model checkpoints, or other third-party dependencies should also be reported
to their respective maintainers.
