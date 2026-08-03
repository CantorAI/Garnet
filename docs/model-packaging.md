# Garnet Model Packaging and Enumeration

CantorOne ships the Garnet runtime and its native dependencies as part of the
application. Model programs and checkpoint weights are installed separately
after the user selects a compatible model from the signed remote catalog:

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
root. `list_loaded_models_json()` reports live serving instances separately
from installed packages. Garnet currently owns one serving instance, so the
response declares `serving_mode: "single_instance"`; the array contract can
remain unchanged when the model manager becomes multi-instance.

Native C++, Python, Electron/JavaScript, and `.x` callers all consume the same
XLang `garnet` package. No Garnet-specific C header is part of the public
integration contract.

## Remote model repository

The Manifold host stores only the signed catalog below `/data/garnetmodels`
and exposes it from `https://garnetmodel.ai/api/v1/garnet/models/`:

```text
/data/garnetmodels/
  catalog-v1.json
  catalog-v1.sig
```

The catalog has four product categories: LLM (`text` in the wire contract),
VLM, ASR, and TTS. Large immutable weight parts are GitHub Release assets in
CantorAI/ModelZoo. Garnet verifies the catalog signature before showing remote
entries, verifies every part and reconstructed file with SHA-256, and activates
the model with an atomic directory rename.

CantorOne never downloads the Garnet runtime. Community applications may get
the same runtime binary from the public Garnet GitHub releases. The repository
also carries public XModel source and examples for Python, C++, Electron, and
`.x`; checkpoint weights are distributed through ModelZoo according to their
upstream and Garnet license terms.
