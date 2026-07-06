# Garnet Workbench

Garnet Workbench is the model-understanding and debug surface for Garnet.

The first target is Qwen3-VL, but the Workbench should stay model-agnostic. It
should read Garnet's `.x` model source, model config, backend lowering metadata,
and optional runtime debug traces, then show how a model becomes executable GPU
work.

## MVP Goal

Build a local web tool that answers these questions quickly:

- What blocks does this model contain?
- Which `.x` function produced each block?
- Which operations are already supported by Garnet?
- Which operations are open ops that need TensorRT, custom CUDA, or CPU/debug handling?
- What tensors flow through vision, multimodal merge, text prefill, KV cache, and decode?
- During debug, where did a tensor snapshot, shape mismatch, or backend mismatch happen?

## MVP Inputs

- `qwen_vl/xmodel/*.x`
- Qwen3-VL `config.json`
- optional weight manifest or safetensors index
- optional backend capability registry
- optional JSONL runtime trace
- optional tensor snapshot metadata

## MVP Outputs

- static model graph JSON
- op support table
- backend partition view
- debug trace overlay
- tensor snapshot index
- missing/open-op checklist

## Run The Current MVP

Open this file in a browser:

```text
workbench/web/index.html
```

The implementation is dependency-free and loads the latest generated Qwen `.x`
graph from `workbench/web/default_graph.js`. It can also load graph JSON or
trace JSONL files from disk.

The page is a fixed SPA shell: the whole browser page should not scroll. Only
components such as the sidebar, graph viewport, op table, detail inspector, and
trace table scroll or pan internally.

The browser app is a small SPA with hash routes:

- `#graph`: pan/zoom graph canvas
- `#ops`: op support table and selected-node detail
- `#trace`: loaded trace events

Graph controls:

- mouse wheel: zoom around cursor
- drag empty canvas: pan
- `Fit`: fit graph to viewport
- `+` / `-`: zoom in or out
- `Back`: return one graph level
- `Overview`: jump to the high-level architecture

Graph drill-down levels:

- high-level model architecture
- subsystem graph such as vision, multimodal merge, text decoder, or matrices
- function/layer graph
- final function graph with calls, loops/branches, ops, and matrices

## Layout Direction

The current MVP uses ELK.js layered layout:

- top-to-bottom ranks for block, control flow, ops/calls, and matrices
- left-to-right dataflow through ELK Layered
- dashed frames for repeat/branch regions
- separate edge styles for execution flow and weight/matrix inputs

ELK is loaded from jsDelivr for the dependency-free MVP. If it cannot load, the
Workbench falls back to the local semantic layered layout.

Generate a graph from the current Qwen `.x` files:

```text
node workbench/tools/extract_xmodel_graph.js --out workbench/examples/qwen3-vl-current.graph.json
```

Generate the default graph used by the SPA:

```text
node workbench/tools/extract_xmodel_graph.js --out workbench/examples/qwen3-vl-current.graph.json --js-out workbench/web/default_graph.js
```

After generating `default_graph.js`, refresh the SPA. Use `Load Graph JSON` only
when you want to inspect a different graph artifact.

## Proposed Folder Layout

```text
workbench/
  README.md
  design/
    mvp.md
    qwen3-vl-mapping.md
  schemas/
    graph.schema.json
    trace.schema.json
  examples/
    qwen3-vl-2b.graph.json
  web/
    index.html
    styles.css
    app.js
  tools/
    extract_xmodel_graph.js
```

The MVP can start as a static generated graph plus a simple browser UI. Later it
can become an interactive app connected to Garnet runtime sessions.
