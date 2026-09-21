# Garnet Model Packaging and Enumeration

Host applications ship the Garnet runtime and install model programs and
checkpoint weights separately. Garnet receives local package paths and does
not own network access, catalog retrieval, or package installation:

```text
CantorOne/resources/garnet/bin/garnet.dll
UserData/garnet/models/qwen3/text_1_7b/
    model.json
    prefill.x
    decode.x
    decode_batch.x
    profiles/*.json
UserData/garnet/models/qwen3/vl_2b_instruct/
    model.json
    *.x
    profiles/*.json
```

The `garnet_xmodels` CMake install component contains only `.x` and `.json`
files. Checkpoint weights, tests, documentation, generated TensorRT engines,
OpenVINO compiled blobs, and native quantized packs are not part of it.
Weights remain external and are identified by the `weights` section of each
`model.json` manifest.

## Catalog discovery

The installed-model catalog is resolved in this order:

1. An explicit catalog path passed by the application.
2. `GARNET_MODEL_CATALOG`.
3. A `models/` directory beside the runtime `bin/` directory.

Every manifest, entrypoint, and profile is validated during enumeration.
Invalid packages remain visible with `available: false` and an `issues` array;
they must not be offered for loading by an application.

## Application APIs

The xlang package exports:

```python
available = garnet.list_available_models_json()
loaded = garnet.list_loaded_models_json()
```

`list_available_models_json(catalog_root)` accepts an optional custom catalog
root. `list_loaded_models_json()` reports the live model instances owned by the
runtime. Local package discovery and installation remain host responsibilities.

Native C++, Python, Electron/JavaScript, and `.x` callers all consume the same
XLang `garnet` package. No Garnet-specific C header is part of the public
integration contract.

## Host-managed packages

The host application is responsible for catalog trust, downloads, signature
and hash verification, installation, updates, and removal. Once a package is
available locally, the host passes its model root, xModel root, cache root, and
model ID to `serve_model(...)`.

Keeping package distribution outside the inference runtime lets Garnet remain
embeddable in products with different stores, security policies, and deployment
topologies. Checkpoint weights retain their upstream license terms.
