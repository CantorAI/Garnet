# Contributing to Garnet

Thank you for helping improve Garnet. Contributions should preserve the central
architecture: xModel programs describe backend-neutral Tensor Expressions, and
backend-specific behavior remains below the graph-lowering boundary.

## Before You Start

- Read the [Getting Started guide](docs/getting-started.md).
- Search existing issues before opening a new one.
- Keep changes focused; separate unrelated cleanup from behavioral work.
- Do not commit model weights, vendor SDKs, generated engines, credentials, or
  media without clear redistribution rights.

## Development Workflow

1. Fork the repository and create a branch from `main`.
2. Build Garnet and run the relevant native and XLang3 tests.
3. Add tests for behavior changes.
4. Update public documentation when an interface or supported path changes.
5. Open a pull request describing the problem, design, validation, hardware,
   backend, precision, and any performance impact.

## xModel Changes

Changes to an xModel program should include:

- source-import and graph-capture coverage;
- explicit backend lowering coverage for every newly reached operator;
- numerical checks before claiming pretrained inference support;
- an updated manifest or backend profile when the public contract changes.

Graph capture proves program structure, not backend correctness. Keep capture,
lowering, pretrained inference, and performance claims distinct.

## Backend and Kernel Changes

- Unsupported operations must fail explicitly.
- Include numerical tests against an independent reference where practical.
- State supported devices, data types, shapes, and alignment requirements.
- Preserve backend-neutral runtime interfaces.
- Report benchmark hardware, software versions, warmup, cache state, batch,
  input/output lengths, average latency, and tail latency.

## Copyright and License

By submitting a contribution, you agree that it may be distributed under the
repository's Apache License 2.0. Preserve applicable third-party notices and do
not relicense external code, model assets, or test media as CantorAI work.
