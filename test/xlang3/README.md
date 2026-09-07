# XLang3 Integration Tests

This suite covers Garnet's XLang3 integration, including the actual native DLL,
backend inference, and serving executable. The focused `garnet_tensor_tests`
target also compiles the real `GarnetTensor` provider and graph-capture adapter
with a test-only package for observing captured graphs.

Build `xlang3_runtime` in the workspace Release build first, then run the existing
Garnet build script with target `garnet_tensor_tests`:

```bat
build_scripts\windows\build_release.bat garnet_tensor_tests
```

The executable uses the workspace `bin/Release` directory. Run it with the
absolute path to `test/xlang3/tensor_graph.py`, or run CTest in the generated
`garnet-tensor-tests` build directory.

Coverage:

- Real C++ package `AddClass` construction of `GarnetTensor`.
- Registered unary/binary expression capture, including tensor-valued keywords.
- Intrinsic CPU expressions mixed with Garnet operations.
- Repeated fusion invocation IDs, region annotations, and output aliases.
- Snapshot semantics for mutable keyword containers.
- Factory lifetime after provider destruction.
- FP16/BF16 symbolic weights, raw storage, and strided views.
- Zero-copy capture and exactly-once external storage cleanup.
- Explicit rejection of unsupported low-precision CPU arithmetic.
- Fusion partition boundaries, repeated atomic regions, and structured outputs.
- CUDA transfers, strided views, and borrowed-allocation ownership on a CUDA device.

The host/runtime must outlive native values and capture objects. The CUDA toolkit
is required to build the storage helper. GPU storage checks run when a CUDA
device is available and print an explicit skip otherwise. These checks do not
prove model inference; the separate native model suites cover that path.

The workspace Garnet CTest directory also runs the full native DLL tests:

- TensorRT and OpenVINO graph compilation, inference, and disk-cache reload.
- Model-source isolation and dependency edits within one process.
- Invalid model/frontend contracts without downloaded checkpoints.
- CPU/GPU image normalization, metadata, and JPEG file/bytes parity.
- Serving process requests, idle lifetime, error recovery, and EOF shutdown.
- Native quantization, AVX2/scalar parity, catalog, scheduler, KV, and sampling.
- Native package classes, writable properties, tensor operations, and invalid inputs.

Model test details are in `models/README.md`. TensorRT tests add the configured
TensorRT SDK's `bin` directory to their process search path; they do not require
copying the entire acceleration SDK beside the application. OpenVINO tests are
registered when its native SDK is configured. Missing CUDA devices fail the
device execution suites; only the tensor-storage probe has an explicit skip.
