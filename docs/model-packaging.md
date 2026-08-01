# Garnet Model Packaging and Enumeration

Garnet releases install production model programs below `models/`, next to
the runtime `bin/` directory:

```text
Garnet/
  bin/garnet.dll
  models/qwen3/text_1_7b/
    model.json
    prefill.x
    decode.x
    decode_batch.x
    profiles/*.json
  models/qwen3/vl_2b_instruct/
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

The available-model catalog is resolved in this order:

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

Native applications can include `garnet/garnet_serving.h` and call:

```c
GarnetListAvailableModelsJson(catalog_root, output, capacity, &required);
GarnetListLoadedModelsJson(output, capacity, &required);
```

The required byte count includes the trailing NUL. A null or undersized output
buffer returns `2` after setting the required capacity, enabling the usual
two-call allocation pattern.
