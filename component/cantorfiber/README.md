# Garnet CantorFiber Component Contract

Garnet stays a generic native inference runtime. CantorFiber owns only the
component contract: it reads `component.json`, resolves package dependencies
from Manifold's signed package catalog, supervises `GarnetServ.exe`, and
routes Portal-discovered `component:/...` APIs over the same process IPC slot
used by other native components.

Manifold stores only signed catalog/manifest JSON. Runtime, acceleration, and
model payloads remain immutable GitHub Release assets referenced by package
catalog entries with exact size and SHA-256 metadata.

The serving executable loads the Garnet XLang package directly with Python
disabled. It accepts newline-delimited JSON requests on stdin and writes one
JSON response per line on stdout:

```json
{"id":"1","path":"component:/v1/chat/completions","body":{"model":"Qwen3-1.7B","messages":[{"role":"user","content":"Hello"}]}}
```

Routes are declared only in `component.json` and the schemas under `schemas/`,
so Portal can render available APIs from component metadata without hard-coded
Garnet UI.

`GarnetServ.exe` is model-agnostic. Requests provide `body.model`, or
CantorFiber may inject `GARNET_DEFAULT_MODEL_ID`,
`GARNET_DEFAULT_VLM_MODEL_ID`, `GARNET_DEFAULT_ASR_MODEL_ID`, or
`GARNET_DEFAULT_TTS_MODEL_ID` from a tenant profile. The executable expects
`xlang_eng.dll` to be staged in its own directory or in `GARNET_XLANG_DIR`.
