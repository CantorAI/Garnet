let graphData = window.GARNET_DEFAULT_GRAPH || {
  schema_version: "0.1",
  model: {
    name: "Qwen3-VL-2B-Instruct",
    family: "qwen3_vl",
    source: "xModel/qwen3/vl_2b_instruct/qwen_vl_model.x",
    config: {
      text_layers: 28,
      text_hidden_size: 2048,
      attention_heads: 16,
      kv_heads: 8,
      head_dim: 128,
      vision_depth: 24,
      vision_hidden_size: 1024,
      vision_heads: 16,
      patch_size: 16,
      temporal_patch_size: 2,
      spatial_merge_size: 2,
      deepstack_visual_indexes: [5, 11, 17]
    }
  },
  nodes: [
    node("model.qwen3_vl", "model", "Qwen3VLModel", "", "", "planned", "xModel/qwen3/vl_2b_instruct/qwen_vl_model.x", "Qwen3VLModel", 19),
    node("vision.patch_embed", "op", "Vision PatchEmbed Conv3d", "qwen3_vl_patch_embed_conv3d", "trt", "planned", "xModel/qwen3/vl_2b_instruct/vision_encoder.x", "Qwen3VisionEncoder", 154),
    node("vision.blocks", "block", "Vision Blocks x24", "", "", "planned", "xModel/qwen3/vl_2b_instruct/vision_encoder.x", "VisionBlock", 103),
    node("vision.attention", "op", "Vision Varlen Attention", "vision_varlen_attention", "cuda", "missing", "xModel/qwen3/vl_2b_instruct/vision_encoder.x", "VisionAttention", 87),
    node("vision.deepstack", "block", "DeepStack Mergers", "", "", "planned", "xModel/qwen3/vl_2b_instruct/vision_encoder.x", "Qwen3VisionEncoder", 179),
    node("adapter.merge_visual", "op", "Visual Placeholder Merge", "qwen3_vl_merge_visual_embeddings", "cuda", "missing", "xModel/qwen3/vl_2b_instruct/vl_adapter.x", "Qwen3MergeVisualEmbeddings", 36),
    node("adapter.mrope_index", "op", "MRoPE Position Builder", "qwen3_vl_get_rope_index", "cuda", "missing", "xModel/qwen3/vl_2b_instruct/vl_adapter.x", "Qwen3GetRopeIndex", 22),
    node("text.layers", "block", "Text Decoder Layers x28", "", "", "planned", "xModel/qwen3/vl_2b_instruct/qwen_llm.x", "Qwen3TextDecoderLayer", 86),
    node("text.paged_kv_update", "op", "Paged KV Update", "paged_kv_update", "cuda", "missing", "xModel/qwen3/vl_2b_instruct/qwen_llm.x", "Qwen3TextAttention", 57),
    node("text.paged_attention", "op", "Paged Attention", "paged_attention", "cuda", "missing", "xModel/qwen3/vl_2b_instruct/qwen_llm.x", "Qwen3TextAttention", 63),
    node("text.lm_head", "op", "LM Head", "lm_head", "trt", "planned", "xModel/qwen3/vl_2b_instruct/qwen_vl_model.x", "Qwen3VLModel", 71)
  ],
  edges: [
    edge("model.qwen3_vl", "vision.patch_embed"),
    edge("vision.patch_embed", "vision.blocks"),
    edge("vision.blocks", "vision.attention"),
    edge("vision.blocks", "vision.deepstack"),
    edge("vision.deepstack", "adapter.merge_visual"),
    edge("adapter.merge_visual", "adapter.mrope_index"),
    edge("adapter.mrope_index", "text.layers"),
    edge("text.layers", "text.paged_kv_update"),
    edge("text.paged_kv_update", "text.paged_attention"),
    edge("text.paged_attention", "text.lm_head")
  ]
};

const state = {
  selectedId: graphData.nodes[0].id,
  selectedGraphItem: null,
  route: "graph",
  detailTab: "layers",
  graphFocus: "overview",
  graphHistory: [],
  query: "",
  enabledStatuses: new Set(),
  traceEvents: [],
  viewport: {
    x: 20,
    y: 20,
    scale: 1
  },
  graphSize: {
    width: 2240,
    height: 460
  },
  lastGraphView: null,
  lastPositions: {},
  layoutToken: 0,
  elk: null,
  pan: {
    active: false,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0
  }
};

const statusColors = {
  defined: "#46c2a5",
  supported: "#85d68a",
  missing: "#ef6f6c",
  unknown: "#b89cff",
  mismatch: "#f07c9d",
  planned: "#7cb7ff"
};

const backendColors = {
  native: "#85d68a",
  trt: "#60c3f0",
  cuda: "#f0b45b",
  plugin: "#46c2a5",
  debug: "#d68cff",
  missing: "#ef6f6c",
  unknown: "#b89cff",
  mismatch: "#f07c9d"
};

const CARD_WIDTH = 236;
const CARD_HEIGHT = 132;
const COL_WIDTH = 282;
const ROW_HEIGHT = 148;

function node(id, kind, label, op, backend, status, file, func, line) {
  return {
    id,
    kind,
    label,
    op,
    backend,
    status,
    source: { file, function: func, line }
  };
}

function edge(from, to, kind = "flow", label = "") {
  return { from, to, kind, label };
}

function init() {
  graphData.nodes.forEach((item) => state.enabledStatuses.add(item.status));
  bindEvents();
  bindMinimapEvents();
  bindSplitters();
  render();
}

function bindEvents() {
  document.querySelectorAll(".route-button").forEach((button) => {
    button.addEventListener("click", () => {
      setRoute(button.dataset.route);
    });
  });

  window.addEventListener("hashchange", () => {
    setRoute(location.hash.replace("#", "") || "graph", false);
  });

  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.detailTab = button.dataset.tab;
      renderDrillDetail();
    });
  });

  document.getElementById("search-input").addEventListener("input", (event) => {
    state.query = event.target.value.trim().toLowerCase();
    render();
  });

  document.getElementById("select-matrices").addEventListener("click", () => {
    graphData.nodes.forEach((item) => state.enabledStatuses.add(item.status));
    state.query = "matrix";
    document.getElementById("search-input").value = "matrix";
    render();
  });

  document.getElementById("select-all").addEventListener("click", () => {
    graphData.nodes.forEach((item) => state.enabledStatuses.add(item.status));
    render();
  });

  document.getElementById("fit-graph").addEventListener("click", () => {
    fitGraph();
  });
  document.getElementById("overview-graph").addEventListener("click", showOverviewGraph);
  document.getElementById("graph-back").addEventListener("click", goBackGraph);
  document.getElementById("graph-root").addEventListener("click", showOverviewGraph);
  document.getElementById("zoom-out").addEventListener("click", () => zoomGraph(0.85));
  document.getElementById("zoom-in").addEventListener("click", () => zoomGraph(1.18));

  document.getElementById("graph-input").addEventListener("change", handleGraphFile);
  document.getElementById("trace-input").addEventListener("change", handleTraceFile);
  document.getElementById("clear-trace").addEventListener("click", () => {
    state.traceEvents = [];
    renderDetail();
  });

  bindCanvasEvents();
}

function render() {
  renderMetrics();
  renderFilters();
  renderGraph();
  renderTable();
  renderDetail();
  renderTraceRoute();
  renderDrillDetail();
  renderSelectionExplain();
  renderRoute();
}

function renderMetrics() {
  const layers = graphData.nodes.filter((item) => item.kind === "block").length;
  const matrices = graphData.nodes.filter((item) => item.kind === "tensor" || item.kind === "matrix").length;
  const knownMatrixParams = graphData.nodes
    .filter((item) => item.kind === "matrix")
    .reduce((sum, item) => sum + (Number.isFinite(item.metadata?.parameter_count) ? item.metadata.parameter_count : 0), 0);
  document.getElementById("model-name").textContent = graphData.model.name;
  document.getElementById("node-count").textContent = graphData.nodes.length;
  document.getElementById("edge-count").textContent = graphData.edges.length;
  document.getElementById("layer-count").textContent = layers;
  document.getElementById("matrix-count").textContent = matrices;
  document.getElementById("matrix-param-count").textContent = formatParamCount(knownMatrixParams);
}

function renderFilters() {
  const root = document.getElementById("status-filters");
  const statuses = [...new Set(graphData.nodes.map((item) => item.status))].sort();
  root.innerHTML = statuses.map((status) => {
    const count = graphData.nodes.filter((item) => item.status === status).length;
    const checked = state.enabledStatuses.has(status) ? "checked" : "";
    return `
      <label class="filter">
        <span><input type="checkbox" data-status="${escapeHtml(status)}" ${checked}>${escapeHtml(status)}</span>
        <small>${count}</small>
      </label>
    `;
  }).join("");

  root.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.addEventListener("change", (event) => {
      const status = event.target.dataset.status;
      if (event.target.checked) {
        state.enabledStatuses.add(status);
      } else {
        state.enabledStatuses.delete(status);
      }
      renderGraph();
      renderTable();
    });
  });
}

function filteredNodes() {
  return graphData.nodes.filter((item) => {
    if (!state.enabledStatuses.has(item.status)) {
      return false;
    }
    if (!state.query) {
      return true;
    }
    const haystack = [
      item.id,
      item.kind,
      item.label,
      item.op,
      item.backend,
      item.status,
      item.source?.file,
      item.source?.function
    ].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(state.query);
  });
}

async function renderGraph() {
  const canvas = document.getElementById("graph-canvas");
  const graphView = buildGraphView();
  const visibleIds = new Set(graphView.nodes.map((item) => item.id));
  const token = ++state.layoutToken;
  const layout = await layoutGraphView(graphView);
  if (token !== state.layoutToken) {
    return;
  }
  const positions = layout.positions;
  const width = layout.width;
  const height = layout.height;
  state.graphSize = { width, height };
  renderGraphTrail();
  document.getElementById("graph-context").textContent = graphContextText(graphView.context);
  document.getElementById("graph-back").disabled = state.graphHistory.length === 0;

  const frames = (graphView.frames || []).map((frame) => {
    const bounds = frame.nodeIds ? boundsForNodes(frame.nodeIds, positions) : null;
    const x = bounds ? bounds.x - 28 : 24 + (frame.col || 0) * COL_WIDTH;
    const y = bounds ? bounds.y - 34 : 20 + (frame.row || 0) * ROW_HEIGHT;
    const w = bounds ? bounds.width + 56 : Math.max(CARD_WIDTH + 44, (frame.cols || 1) * COL_WIDTH - 28);
    const h = bounds ? bounds.height + 68 : Math.max(CARD_HEIGHT + 42, (frame.rows || 1) * ROW_HEIGHT - 34);
    return `
      <g class="graph-frame">
        <rect x="${x}" y="${y}" width="${w}" height="${h}"></rect>
        <text x="${x + 14}" y="${y + 22}">${escapeSvg(frame.label || "region")}</text>
      </g>
    `;
  }).join("");

  const edges = graphView.edges.map((item) => {
    const from = positions[item.from];
    const to = positions[item.to];
    if (!from || !to) {
      return "";
    }
    const dimmed = !visibleIds.has(item.from) || !visibleIds.has(item.to);
    const x1 = from.x + CARD_WIDTH;
    const y1 = from.y + CARD_HEIGHT / 2;
    const x2 = to.x;
    const y2 = to.y + CARD_HEIGHT / 2;
    const routed = layout.edgeSections[item.from + "->" + item.to];
    const d = routed || `M ${x1} ${y1} C ${x1 + 60} ${y1}, ${x2 - 60} ${y2}, ${x2} ${y2}`;
    const label = item.label ? edgeLabelSvg(item.label, x1, y1, x2, y2, dimmed) : "";
    return `<path class="edge ${escapeHtml(item.kind || "")}" opacity="${dimmed ? "0.18" : "1"}" d="${d}"></path>${label}`;
  }).join("");

  const cards = graphView.nodes.map((item) => {
    const pos = positions[item.id];
    const selected = item.id === state.selectedId ? "selected" : "";
    const dimmed = visibleIds.has(item.id) ? "" : "dimmed";
    const color = statusColors[item.status] || "#7cb7ff";
    const backendColor = backendColors[item.backend] || color;
    const sub = nodeSubtitle(item);
    const labelLines = svgTextLines(item.label, 16, 38, 16, 28, "", 4);
    const subLines = svgTextLines(sub, 16, 92, 13, 36, "sub", item.kind === "matrix" ? 3 : 2);
    return `
      <g class="node-card ${selected} ${dimmed}" data-node-id="${escapeHtml(item.id)}" transform="translate(${pos.x}, ${pos.y})">
        <title>${escapeHtml(nodeTooltip(item))}</title>
        <rect width="${CARD_WIDTH}" height="${CARD_HEIGHT}"></rect>
        <circle class="status-dot" cx="16" cy="18" r="5" fill="${color}"></circle>
        <circle class="status-dot" cx="${CARD_WIDTH - 16}" cy="18" r="5" fill="${backendColor}"></circle>
        ${labelLines}
        ${subLines}
      </g>
    `;
  }).join("");

  canvas.innerHTML = `
    <svg class="graph-svg" viewBox="0 0 ${width} ${height}" role="img">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M 0 0 L 10 4 L 0 8 z" fill="#55606d"></path>
        </marker>
      </defs>
      <g id="graph-viewport">
        ${frames}
        <g marker-end="url(#arrow)">${edges}</g>
        ${cards}
      </g>
    </svg>
    <div id="minimap" class="minimap" aria-label="Graph minimap"></div>
  `;
  state.lastGraphView = graphView;
  state.lastPositions = positions;
  applyViewport();
  renderMinimap();

  canvas.querySelectorAll(".node-card").forEach((card) => {
    card.addEventListener("click", () => {
      handleGraphNodeClick(card.dataset.nodeId, graphView);
      renderGraph();
      renderTable();
      renderDetail();
      renderDrillDetail();
      renderSelectionExplain();
    });
  });
}

async function layoutGraphView(graphView) {
  if (window.ELK) {
    try {
      return await layoutGraphViewWithElk(graphView);
    } catch (error) {
      console.warn("ELK layout failed, using fallback layout.", error);
    }
  }
  return layoutGraphViewFallback(graphView.nodes);
}

async function layoutGraphViewWithElk(graphView) {
  if (!state.elk) {
    state.elk = new ELK();
  }
  const elkGraph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.spacing.nodeNode": "48",
      "elk.layered.spacing.nodeNodeBetweenLayers": "90",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.layered.nodePlacement.favorStraightEdges": "true"
    },
    children: graphView.nodes.map((item) => ({
      id: item.id,
      width: CARD_WIDTH,
      height: CARD_HEIGHT,
      layoutOptions: {
        "elk.layered.layering.layerConstraint": item.metadata?.elkLayer || "NONE"
      }
    })),
    edges: graphView.edges.map((item, index) => ({
      id: `e${index}`,
      sources: [item.from],
      targets: [item.to]
    }))
  };

  const laidOut = await state.elk.layout(elkGraph);
  const positions = {};
  (laidOut.children || []).forEach((item) => {
    positions[item.id] = { x: (item.x || 0) + 42, y: (item.y || 0) + 42 };
  });
  const edgeSections = {};
  (laidOut.edges || []).forEach((item, index) => {
    const source = graphView.edges[index]?.from;
    const target = graphView.edges[index]?.to;
    const section = item.sections?.[0];
    if (!source || !target || !section) {
      return;
    }
    edgeSections[source + "->" + target] = elkSectionToPath(section, 42, 42);
  });

  const maxX = Math.max(...Object.values(positions).map((pos) => pos.x + CARD_WIDTH), CARD_WIDTH);
  const maxY = Math.max(...Object.values(positions).map((pos) => pos.y + CARD_HEIGHT), CARD_HEIGHT);
  return {
    positions,
    edgeSections,
    width: Math.max(1200, maxX + 90),
    height: Math.max(540, maxY + 90)
  };
}

function elkSectionToPath(section, offsetX, offsetY) {
  const points = [
    section.startPoint,
    ...(section.bendPoints || []),
    section.endPoint
  ].filter(Boolean).map((point) => ({ x: point.x + offsetX, y: point.y + offsetY }));
  if (points.length < 2) {
    return "";
  }
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
}

function layoutGraphViewFallback(nodes) {
  const positions = {};
  nodes.forEach((item, index) => {
    const col = item.metadata?.col ?? index;
    const row = item.metadata?.row ?? laneForKind(item.kind);
    positions[item.id] = { x: 42 + col * COL_WIDTH, y: 42 + row * ROW_HEIGHT };
  });
  const width = Math.max(1200, Math.max(...Object.values(positions).map((pos) => pos.x)) + CARD_WIDTH + 90);
  const height = Math.max(540, Math.max(...Object.values(positions).map((pos) => pos.y)) + CARD_HEIGHT + 80);
  return { positions, edgeSections: {}, width, height };
}

function laneForKind(kind) {
  if (kind === "model") return 0;
  if (kind === "block") return 0;
  if (kind === "op") return 1;
  if (kind === "runtime") return 1;
  if (kind === "matrix" || kind === "tensor") return 2;
  return 1;
}

function boundsForNodes(nodeIds, positions) {
  const points = nodeIds.map((id) => positions[id]).filter(Boolean);
  if (!points.length) {
    return null;
  }
  const minX = Math.min(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxX = Math.max(...points.map((point) => point.x + CARD_WIDTH));
  const maxY = Math.max(...points.map((point) => point.y + CARD_HEIGHT));
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  };
}

function buildGraphView() {
  if (state.graphFocus === "overview") {
    return buildOverviewGraph();
  }
  if (state.graphFocus.startsWith("subsystem:")) {
    return buildSubsystemGraph(state.graphFocus.replace("subsystem:", ""));
  }
  if (state.graphFocus.startsWith("function:")) {
    return buildFunctionGraph(state.graphFocus.replace("function:", ""));
  }
  return buildOverviewGraph();
}

function renderGraphTrail() {
  const root = document.getElementById("graph-trail");
  if (!root) {
    return;
  }
  const items = graphTrailItems(state.graphFocus);
  root.innerHTML = items.map((item, index) => {
    const current = item.focus === state.graphFocus ? " current" : "";
    const separator = index === 0 ? "" : `<span class="graph-separator">/</span>`;
    return `
      ${separator}
      <button class="graph-crumb${current}" type="button" data-focus="${escapeHtml(item.focus)}" ${current ? "aria-current=\"page\"" : ""}>
        ${escapeHtml(item.label)}
      </button>
    `;
  }).join("");
  root.querySelectorAll(".graph-crumb").forEach((button) => {
    button.addEventListener("click", () => {
      jumpGraphFocus(button.dataset.focus);
    });
  });
}

function graphTrailItems(focus) {
  const items = [{ label: "Overview", focus: "overview" }];
  if (focus === "overview") {
    return items;
  }
  if (focus.startsWith("subsystem:")) {
    const subsystem = focus.replace("subsystem:", "");
    items.push({ label: subsystemLabel(subsystem), focus });
    return items;
  }
  if (focus.startsWith("function:")) {
    const functionName = focus.replace("function:", "");
    const subsystem = subsystemForFunction(functionName);
    if (subsystem) {
      items.push({ label: subsystemLabel(subsystem), focus: `subsystem:${subsystem}` });
    }
    items.push({ label: functionName, focus });
  }
  return items;
}

function jumpGraphFocus(nextFocus) {
  if (!nextFocus || nextFocus === state.graphFocus) {
    return;
  }
  const trail = graphTrailItems(nextFocus);
  state.graphHistory = trail.slice(0, -1).map((item) => item.focus);
  state.graphFocus = nextFocus;
  state.viewport = { x: 20, y: 20, scale: 1 };
  setRoute("graph");
  renderGraph();
}

function subsystemForFunction(functionName) {
  const item = graphData.nodes.find((nodeItem) => nodeItem.source?.function === functionName);
  if (!item) {
    return "";
  }
  for (const subsystem of ["model", "vision", "adapter", "text", "matrix"]) {
    if (subsystem !== "matrix" && belongsToSubsystem(item, subsystem)) {
      return subsystem;
    }
  }
  return "";
}

function subsystemLabel(subsystem) {
  const labels = {
    model: "Model",
    vision: "Vision Tower",
    adapter: "Multimodal Merge",
    text: "Text Decoder",
    matrix: "Weights / Matrices"
  };
  return labels[subsystem] || titleCase(subsystem);
}

function buildOverviewGraph() {
  const matrixCount = graphData.nodes.filter((item) => item.kind === "matrix").length;
  const knownMatrixParams = graphData.nodes
    .filter((item) => item.kind === "matrix")
    .reduce((sum, item) => sum + (Number.isFinite(item.metadata?.parameter_count) ? item.metadata.parameter_count : 0), 0);
  const templateParamTip = [
    "Template-known params only.",
    "Calculated as sum(parameter_count) over extracted matrix template nodes.",
    "Repeated Qwen3-VL layers are not expanded yet.",
    "Full model estimate should multiply repeated vision/text layer matrices by config layer counts."
  ].join(" ");
  const nodes = [
    viewNode("overview.input", "model", "Inputs", "image/video + text", "subsystem:model", 0, 0),
    viewNode("overview.vision", "block", "Vision Tower", "patch embed, vision blocks, merger", "subsystem:vision", 1, 0),
    viewNode("overview.adapter", "block", "Multimodal Merge", "visual placeholders + MRoPE ids", "subsystem:adapter", 2, 0),
    viewNode("overview.text", "block", "Text Decoder", "decoder layers, KV, attention", "subsystem:text", 3, 0),
    {
      ...viewNode("overview.matrices", "matrix", "Weights / Matrices", "projection, norm, embedding weights", "subsystem:matrix", 2, 1),
      metadata: {
        focus: "subsystem:matrix",
        col: 2,
        row: 1,
        summary: `${matrixCount} matrices | template ${formatParamCount(knownMatrixParams)}`,
        tooltip: templateParamTip
      }
    },
    viewNode("overview.output", "model", "Logits", "lm head output", "subsystem:model", 4, 0)
  ];
  const edges = [
    edge("overview.input", "overview.vision"),
    edge("overview.vision", "overview.adapter"),
    edge("overview.adapter", "overview.text"),
    edge("overview.text", "overview.output"),
    edge("overview.matrices", "overview.vision"),
    edge("overview.matrices", "overview.adapter"),
    edge("overview.matrices", "overview.text")
  ];
  return { title: "Overview: Qwen3-VL architecture", nodes, edges };
}

function buildSubsystemGraph(subsystem) {
  const nodes = graphData.nodes
    .filter((item) => belongsToSubsystem(item, subsystem))
    .filter((item) => item.kind === "block")
    .filter((item) => isMajorBlock(item))
    .map((item, index) => ({
      ...item,
      metadata: { ...(item.metadata || {}), focus: `function:${item.source?.function || item.label}`, col: index % 5, row: Math.floor(index / 5) }
    }));

  const edges = [];
  for (let i = 0; i < nodes.length - 1; i += 1) {
    edges.push(edge(nodes[i].id, nodes[i + 1].id));
  }

  if (subsystem === "matrix") {
    return buildMatrixSubsystemGraph();
  }

  return {
    title: `${titleCase(subsystem)} subsystem: blocks/functions`,
    nodes: nodes.length ? nodes : [viewNode(`empty.${subsystem}`, "block", "No blocks", "No matching blocks in graph", "overview", 0, 0)],
    edges
  };
}

function buildFunctionGraph(functionName) {
  const block = graphData.nodes.find((item) => item.kind === "block" && item.source?.function === functionName);
  const formulaGraph = buildKnownFormulaGraph(functionName, block);
  if (formulaGraph) {
    return formulaGraph;
  }
  const ops = graphData.nodes.filter((item) => item.kind === "op" && item.source?.function === functionName);
  const matrices = collectMatricesForFunction(functionName, { includeCalled: false });
  const controls = extractControlNodes(block, functionName);
  const calls = extractFunctionCalls(block, functionName);
  const hasControl = controls.length > 0;
  const consumers = [...calls, ...ops];
  const stages = buildFunctionStages(controls, consumers, matrices);
  const nodes = [];
  stages.forEach((stage) => {
    stage.nodes.forEach((item) => nodes.push(item));
  });

  const edges = [];
  const stageHeads = stages.map((stage) => stage.primary).filter(Boolean);
  if (stageHeads.length) {
    for (let i = 0; i < stageHeads.length - 1; i += 1) {
      edges.push(edge(stageHeads[i].id, stageHeads[i + 1].id, "flow"));
    }
  }
  matrices.forEach((matrix) => {
    const targets = consumers.filter((consumer) => matrixFeedsConsumer(matrix, consumer));
    targets.forEach((target) => edges.push(edge(matrix.id, target.id, "weight")));
  });

  const frames = hasControl
    ? [{
        label: "repeat / branch region",
        nodeIds: [...controls, ...consumers, ...matrices].map((item) => item.id),
        col: 1,
        row: 1,
        cols: Math.max(1, stages.length),
        rows: matrices.length ? 3 : 2
      }]
    : [];

  return {
    title: `Function/layer: ${functionName}`,
    context: block ? {
      label: block.label,
      kind: block.kind,
      source: block.source,
      repeat: repeatInfoForFunction(functionName),
      formula: plainFormulaSummary(block)
    } : null,
    nodes: nodes.length ? nodes : [viewNode(`empty.${functionName}`, "runtime", "No extracted dataflow", "Function scope has no extracted ops, calls, or matrices yet", "overview", 0, 0)],
    edges,
    frames
  };
}

function buildKnownFormulaGraph(functionName, block) {
  if (functionName === "VisionMLP") {
    return buildVisionMlpGraph(functionName, block);
  }
  if (functionName === "Qwen3TextMLP") {
    return buildTextMlpGraph(functionName, block);
  }
  return null;
}

function buildVisionMlpGraph(functionName, block) {
  const matrices = collectMatricesForFunction(functionName, { includeCalled: false });
  const byRole = (pattern) => matrices.find((item) => pattern.test(`${item.id} ${item.label} ${item.metadata?.role || ""}`));
  const x = formulaNode(functionName, "input_x", "input x", "activation", 0, 1, block, {
    summary: "Input activation for one VisionMLP template instance.",
    dims: "[tokens, 1024]",
    formula: "\\(x \\in \\mathbb{R}^{T \\times 1024}\\)"
  });
  const fc1 = formulaNode(functionName, "fc1_linear", "fc1 linear", "matmul + bias", 1, 1, block, {
    summary: "First MLP projection. Uses fc1 weight and bias.",
    dims: "[tokens, 4096]",
    formula: "\\(h_1 = x W_{fc1}^{T} + b_{fc1}\\)"
  });
  const gelu = formulaNode(functionName, "gelu", "GELU", "activation", 2, 1, block, {
    summary: "Vision MLP nonlinearity.",
    dims: "[tokens, 4096]",
    formula: "\\(h_2 = \\mathrm{GELU}(h_1)\\)"
  });
  const fc2 = formulaNode(functionName, "fc2_linear", "fc2 linear", "matmul + bias", 3, 1, block, {
    summary: "Second MLP projection back to hidden size.",
    dims: "[tokens, 1024]",
    formula: "\\(y = h_2 W_{fc2}^{T} + b_{fc2}\\)"
  });
  const y = formulaNode(functionName, "output_y", "output y", "activation", 4, 1, block, {
    summary: "VisionMLP output activation.",
    dims: "[tokens, 1024]",
    formula: "\\(y \\in \\mathbb{R}^{T \\times 1024}\\)"
  });

  const nodes = [
    x,
    withPosition(byRole(/linear_fc1\.weight|fc1.*weight/i), 1, 0),
    withPosition(byRole(/linear_fc1\.bias|fc1.*bias/i), 1, 2),
    fc1,
    gelu,
    withPosition(byRole(/linear_fc2\.weight|fc2.*weight/i), 3, 0),
    withPosition(byRole(/linear_fc2\.bias|fc2.*bias/i), 3, 2),
    fc2,
    y
  ].filter(Boolean);

  return formulaGraphView(functionName, block, nodes, [
    edge(x.id, fc1.id, "activation", "x [T,1024]"),
    edge(nodeId(byRole(/linear_fc1\.weight|fc1.*weight/i)), fc1.id, "weight", "W_fc1 [4096,1024]"),
    edge(nodeId(byRole(/linear_fc1\.bias|fc1.*bias/i)), fc1.id, "weight", "b_fc1 [4096]"),
    edge(fc1.id, gelu.id, "activation", "h1 [T,4096]"),
    edge(gelu.id, fc2.id, "activation", "h2 [T,4096]"),
    edge(nodeId(byRole(/linear_fc2\.weight|fc2.*weight/i)), fc2.id, "weight", "W_fc2 [1024,4096]"),
    edge(nodeId(byRole(/linear_fc2\.bias|fc2.*bias/i)), fc2.id, "weight", "b_fc2 [1024]"),
    edge(fc2.id, y.id, "activation", "y [T,1024]")
  ].filter((item) => item.from && item.to));
}

function buildTextMlpGraph(functionName, block) {
  const matrices = collectMatricesForFunction(functionName, { includeCalled: false });
  const byRole = (pattern) => matrices.find((item) => pattern.test(`${item.id} ${item.label} ${item.metadata?.role || ""}`));
  const x = formulaNode(functionName, "input_x", "input x", "activation", 0, 2, block, {
    summary: "Input activation for one text decoder MLP template instance.",
    dims: "[tokens, 2048]",
    formula: "\\(x \\in \\mathbb{R}^{T \\times 2048}\\)"
  });
  const gate = formulaNode(functionName, "gate_proj", "gate proj", "matmul", 1, 1, block, {
    summary: "Gate projection branch.",
    dims: "[tokens, 6144]",
    formula: "\\(g = x W_{gate}^{T}\\)"
  });
  const silu = formulaNode(functionName, "silu", "SiLU", "activation", 2, 1, block, {
    summary: "Gating activation.",
    dims: "[tokens, 6144]",
    formula: "\\(\\hat{g} = \\mathrm{SiLU}(g)\\)"
  });
  const up = formulaNode(functionName, "up_proj", "up proj", "matmul", 1, 3, block, {
    summary: "Up projection branch.",
    dims: "[tokens, 6144]",
    formula: "\\(u = x W_{up}^{T}\\)"
  });
  const mul = formulaNode(functionName, "gate_mul", "gate multiply", "elementwise multiply", 3, 2, block, {
    summary: "Elementwise gated MLP activation.",
    dims: "[tokens, 6144]",
    formula: "\\(m = \\mathrm{SiLU}(g) \\odot u\\)"
  });
  const down = formulaNode(functionName, "down_proj", "down proj", "matmul", 4, 2, block, {
    summary: "Down projection back to hidden size.",
    dims: "[tokens, 2048]",
    formula: "\\(y = m W_{down}^{T}\\)"
  });
  const y = formulaNode(functionName, "output_y", "output y", "activation", 5, 2, block, {
    summary: "Qwen3TextMLP output activation.",
    dims: "[tokens, 2048]",
    formula: "\\(y \\in \\mathbb{R}^{T \\times 2048}\\)"
  });

  const gateWeight = byRole(/gate_proj.*weight/i);
  const upWeight = byRole(/up_proj.*weight/i);
  const downWeight = byRole(/down_proj.*weight/i);
  const nodes = [
    x,
    withPosition(gateWeight, 1, 0),
    gate,
    silu,
    withPosition(upWeight, 1, 4),
    up,
    mul,
    withPosition(downWeight, 4, 1),
    down,
    y
  ].filter(Boolean);

  return formulaGraphView(functionName, block, nodes, [
    edge(x.id, gate.id, "activation", "x [T,2048]"),
    edge(nodeId(gateWeight), gate.id, "weight", "W_gate [6144,2048]"),
    edge(gate.id, silu.id, "activation", "g [T,6144]"),
    edge(x.id, up.id, "activation", "x [T,2048]"),
    edge(nodeId(upWeight), up.id, "weight", "W_up [6144,2048]"),
    edge(silu.id, mul.id, "activation", "SiLU(g) [T,6144]"),
    edge(up.id, mul.id, "activation", "u [T,6144]"),
    edge(mul.id, down.id, "activation", "m [T,6144]"),
    edge(nodeId(downWeight), down.id, "weight", "W_down [2048,6144]"),
    edge(down.id, y.id, "activation", "y [T,2048]")
  ].filter((item) => item.from && item.to));
}

function formulaGraphView(functionName, block, nodes, edges) {
  return {
    title: `Function/layer: ${functionName}`,
    context: block ? {
      label: block.label,
      kind: block.kind,
      source: block.source,
      repeat: repeatInfoForFunction(functionName),
      formula: plainFormulaSummary(block)
    } : null,
    nodes,
    edges,
    frames: [{
      label: "formula dataflow",
      nodeIds: nodes.map((item) => item.id),
      col: 0,
      row: 0,
      cols: 6,
      rows: 5
    }]
  };
}

function formulaNode(functionName, suffix, label, op, col, row, block, metadata = {}) {
  return {
    id: `formula.${slugVariable(functionName)}.${suffix}`,
    kind: "op",
    label,
    op,
    backend: "native",
    status: "defined",
    source: block?.source,
    metadata: {
      focus: "",
      col,
      row,
      role: "formula",
      ...metadata
    }
  };
}

function withPosition(item, col, row) {
  if (!item) {
    return null;
  }
  return {
    ...item,
    metadata: {
      ...(item.metadata || {}),
      col,
      row
    }
  };
}

function nodeId(item) {
  return item?.id || "";
}

function buildFunctionStages(controls, consumers, matrices) {
  const stages = [];
  controls.forEach((item) => stages.push({ primary: item, items: [item] }));
  consumers.forEach((consumer) => {
    const attachedMatrices = matrices.filter((matrix) => matrixFeedsConsumer(matrix, consumer));
    stages.push({ primary: consumer, items: [consumer, ...attachedMatrices] });
  });
  const attachedIds = new Set(stages.flatMap((stage) => stage.items.map((item) => item.id)));
  matrices.filter((matrix) => !attachedIds.has(matrix.id)).forEach((matrix) => {
    stages.push({ primary: matrix, items: [matrix] });
  });

  return stages.map((stage, stageIndex) => {
    const col = stageIndex + 1;
    const primaryRow = stage.primary.kind === "runtime" ? 1 : 2;
    const nodes = stage.items.map((item, itemIndex) => {
      const row = item.id === stage.primary.id ? primaryRow : primaryRow + 1 + itemIndex;
      return {
        ...item,
        metadata: {
          ...(item.metadata || {}),
          col,
          row
        }
      };
    });
    return { ...stage, nodes };
  });
}

function findMatrixConsumerIndex(matrix, consumers) {
  return consumers.findIndex((consumer) => matrixFeedsConsumer(matrix, consumer));
}

function matrixFeedsConsumer(matrix, consumer) {
  if (consumer.kind === "block") {
    if (matrix.source?.function === consumer.label) {
      return true;
    }
    if (matrix.source?.function === consumer.source?.function && consumer.label === "linear") {
      return /matrix|projection|mlp|attention|kernel|embedding/i.test(matrix.metadata?.role || "");
    }
    if (matrix.source?.function === consumer.source?.function && /norm/i.test(consumer.label)) {
      return /norm/i.test(matrix.metadata?.role || matrix.label || "");
    }
    return false;
  }
  return matrix.source?.line === consumer.source?.line;
}

function extractControlNodes(block, functionName) {
  const expression = block?.metadata?.expression || "";
  const controls = [];
  expression.split(/\r?\n/).forEach((line, index) => {
    const trimmed = line.trim();
    const match = trimmed.match(/^(for|if|elif|else)\b(.*):\s*$/);
    if (!match) {
      return;
    }
    const kind = match[1];
    const condition = kind === "else" ? "else" : match[2].trim();
    controls.push({
      id: `control.${slugVariable(functionName)}.${index + 1}`,
      kind: "runtime",
      label: `${kind}${condition ? ` ${condition}` : ""}`,
      op: kind === "for" ? "loop" : "branch",
      status: "defined",
      source: block?.source,
      metadata: {
        expression: trimmed
      }
    });
  });
  return controls;
}

function extractFunctionCalls(block, functionName) {
  const expression = block?.metadata?.expression || "";
  const localFunctions = new Set(graphData.nodes.filter((item) => item.kind === "block").map((item) => item.source?.function || item.label));
  const calls = [];
  const seen = new Set();
  for (const match of expression.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)) {
    const name = match[1];
    if (name === functionName || !localFunctions.has(name) || seen.has(name)) {
      continue;
    }
    seen.add(name);
    calls.push({
      id: `call.${slugVariable(functionName)}.${slugVariable(name)}`,
      kind: "block",
      label: name,
      op: "function call",
      status: "defined",
      source: block?.source,
      metadata: {
        focus: `function:${name}`,
        expression: `${name}(...)`
      }
    });
  }
  return calls;
}

function collectMatricesForFunction(functionName, options = { includeCalled: true }) {
  const direct = graphData.nodes.filter((item) => item.kind === "matrix" && item.source?.function === functionName);
  if (!options.includeCalled) {
    return uniqueNodes(direct);
  }
  const block = graphData.nodes.find((item) => item.kind === "block" && item.source?.function === functionName);
  const callNames = extractFunctionCalls(block, functionName).map((item) => item.label);
  const viaCalls = graphData.nodes.filter((item) => item.kind === "matrix" && callNames.includes(item.source?.function));
  return uniqueNodes([...direct, ...viaCalls]);
}

function buildMatrixSubsystemGraph() {
  const matrices = graphData.nodes.filter((item) => item.kind === "matrix");
  const relatedFunctions = uniqueNodes(matrices.map((matrix) => {
    return graphData.nodes.find((item) => item.kind === "block" && item.source?.function === matrix.source?.function);
  }).filter(Boolean));
  const relatedOps = uniqueNodes(matrices.flatMap((matrix) => {
    return graphData.nodes.filter((item) => {
      return item.kind === "op" && (
        item.source?.function === matrix.source?.function ||
        item.source?.line === matrix.source?.line
      );
    });
  }));

  const nodes = [];
  relatedFunctions.forEach((item, index) => {
    nodes.push({
      ...item,
      metadata: { ...(item.metadata || {}), focus: `function:${item.source?.function || item.label}`, col: index, row: 0 }
    });
  });
  relatedOps.forEach((item, index) => {
    nodes.push({
      ...item,
      metadata: { ...(item.metadata || {}), col: index, row: 1 }
    });
  });
  matrices.forEach((item, index) => {
    const targetOpIndex = relatedOps.findIndex((op) => op.source?.function === item.source?.function || op.source?.line === item.source?.line);
    const targetFunctionIndex = relatedFunctions.findIndex((fn) => fn.source?.function === item.source?.function);
    nodes.push({
      ...item,
      metadata: {
        ...(item.metadata || {}),
        col: targetOpIndex >= 0 ? targetOpIndex : Math.max(0, targetFunctionIndex),
        row: targetOpIndex >= 0 ? 2 : 1 + Math.floor(index / 4)
      }
    });
  });

  const edges = [];
  relatedFunctions.forEach((fn) => {
    relatedOps
      .filter((op) => op.source?.function === fn.source?.function)
      .forEach((op) => edges.push(edge(fn.id, op.id, "call")));
  });
  matrices.forEach((matrix) => {
    const ops = relatedOps.filter((op) => op.source?.function === matrix.source?.function || op.source?.line === matrix.source?.line);
    if (ops.length) {
      ops.forEach((op) => edges.push(edge(matrix.id, op.id, "weight")));
    } else {
      const fn = relatedFunctions.find((item) => item.source?.function === matrix.source?.function);
      if (fn) {
        edges.push(edge(matrix.id, fn.id, "weight"));
      }
    }
  });

  return {
    title: "Weights / matrices with related ops",
    nodes: nodes.length ? nodes : [viewNode("empty.matrix", "matrix", "No matrices", "No matrix nodes in graph", "overview", 0, 0)],
    edges
  };
}

function belongsToSubsystem(item, subsystem) {
  const text = `${item.id} ${item.source?.file || ""} ${item.source?.function || ""}`.toLowerCase();
  if (subsystem === "model") return text.includes("qwen_vl_model");
  if (subsystem === "vision") return text.includes("vision");
  if (subsystem === "adapter") return text.includes("adapter") || text.includes("merge") || text.includes("rope");
  if (subsystem === "text") return text.includes("qwen_llm") || text.includes("text");
  if (subsystem === "matrix") return item.kind === "matrix";
  return true;
}

function isMajorBlock(item) {
  const name = item.source?.function || item.label || "";
  const helperNames = new Set([
    "linear",
    "layer_norm",
    "rms_norm",
    "gelu",
    "node",
    "edge"
  ]);
  if (helperNames.has(name)) {
    return false;
  }
  return /Qwen|Vision|Attention|MLP|Decoder|Merger|Merge|Rope|Model|Prepare|Patch/i.test(name);
}

function viewNode(id, kind, label, op, focus, col, row) {
  return {
    id,
    kind,
    label,
    op,
    status: "defined",
    metadata: { focus, col, row }
  };
}

function handleGraphNodeClick(nodeId, graphView) {
  const viewItem = graphView.nodes.find((item) => item.id === nodeId);
  if (!viewItem) {
    return;
  }
  state.selectedGraphItem = viewItem;
  state.selectedId = nodeId.startsWith("overview.") ? state.selectedId : nodeId;
  const focus = viewItem.metadata?.focus;
  if (focus && focus !== "overview") {
    navigateGraph(focus);
    return;
  }
  if (viewItem.kind === "block" && viewItem.source?.function) {
    navigateGraph(`function:${viewItem.source.function}`);
    return;
  }
  state.detailTab = viewItem.kind === "op" || viewItem.kind === "matrix" ? "expression" : "layers";
}

function showOverviewGraph() {
  state.graphHistory = [];
  state.graphFocus = "overview";
  state.viewport = { x: 20, y: 20, scale: 1 };
  setRoute("graph");
  renderGraph();
}

function navigateGraph(nextFocus) {
  if (nextFocus === state.graphFocus) {
    return;
  }
  state.graphHistory.push(state.graphFocus);
  state.graphFocus = nextFocus;
  state.viewport = { x: 20, y: 20, scale: 1 };
  setRoute("graph");
}

function goBackGraph() {
  const previous = state.graphHistory.pop();
  if (!previous) {
    return;
  }
  state.graphFocus = previous;
  state.viewport = { x: 20, y: 20, scale: 1 };
  setRoute("graph");
  renderGraph();
}

function renderTable() {
  const tbody = document.getElementById("node-table");
  const rows = filteredNodes();
  document.getElementById("table-count").textContent = `${rows.length} rows`;
  tbody.innerHTML = rows.map((item) => {
    const source = item.source ? `${item.source.file}:${item.source.line}` : "";
    const selected = item.id === state.selectedId ? "background:#18242d" : "";
    return `
      <tr data-node-id="${escapeHtml(item.id)}" style="${selected}">
        <td>${escapeHtml(item.label)}<br><span class="pill">${escapeHtml(item.kind)}</span></td>
        <td>${escapeHtml(item.op || "-")}</td>
        <td><span class="pill">${escapeHtml(item.backend || "-")}</span></td>
        <td><span class="pill">${escapeHtml(item.status)}</span></td>
        <td>${escapeHtml(source)}</td>
      </tr>
    `;
  }).join("");

  tbody.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedId = row.dataset.nodeId;
      state.selectedGraphItem = findNode(state.selectedId);
      renderGraph();
      renderTable();
      renderDetail();
      renderSelectionExplain();
    });
  });
}

function renderDetail() {
  const detail = document.getElementById("node-detail");
  const item = graphData.nodes.find((nodeItem) => nodeItem.id === state.selectedId) || graphData.nodes[0];
  const source = item.source ? `${item.source.file}:${item.source.line}` : "-";
  const traceEvents = state.traceEvents.filter((event) => event.node_id === item.id);
  document.getElementById("selected-id").textContent = item.id;

  const traceHtml = traceEvents.length
    ? `
      <div class="trace-list">
        <h4>Trace Events</h4>
        ${traceEvents.map((event) => `
          <div class="trace-event">
            <span>${escapeHtml(String(event.ts_us || 0))} us</span>
            <span>${escapeHtml(event.event || "event")} ${event.duration_us ? `(${escapeHtml(String(event.duration_us))} us)` : ""}</span>
          </div>
        `).join("")}
      </div>
    `
    : "";

  detail.innerHTML = `
    <dl>
      <dt>ID</dt><dd>${escapeHtml(item.id)}</dd>
      <dt>Label</dt><dd>${escapeHtml(item.label)}</dd>
      <dt>Kind</dt><dd>${escapeHtml(item.kind)}</dd>
      <dt>Op</dt><dd>${escapeHtml(item.op || "-")}</dd>
      <dt>Backend</dt><dd>${escapeHtml(item.backend || "-")}</dd>
      <dt>Status</dt><dd>${escapeHtml(item.status)}</dd>
      <dt>Function</dt><dd>${escapeHtml(item.source?.function || "-")}</dd>
      <dt>Source</dt><dd>${escapeHtml(source)}</dd>
    </dl>
    ${traceHtml}
  `;
}

function renderSelectionExplain() {
  const root = document.getElementById("selection-explain");
  if (!root) {
    return;
  }
  const item = state.selectedGraphItem || getSelectedNode();
  if (!item) {
    root.innerHTML = `<div class="selection-body">Click a graph node to inspect what it represents.</div>`;
    return;
  }
  const source = item.source ? `${item.source.file}:${item.source.line}` : "-";
  const dims = item.kind === "matrix" ? formatNodeDimensions(item) : "";
  const params = item.kind === "matrix" ? formatMatrixParamScope(item) : "";
  const formula = plainFormulaSummary(item);
  const mathFormula = buildMathFormula(item);
  root.innerHTML = `
    <div class="selection-title">${escapeHtml(item.label)}</div>
    <div class="selection-kind">
      <span class="pill">${escapeHtml(item.kind)}</span>
      ${item.op ? `<span class="pill">${escapeHtml(item.op)}</span>` : ""}
      ${item.metadata?.role ? `<span class="pill">${escapeHtml(item.metadata.role)}</span>` : ""}
    </div>
    <div class="selection-body">${escapeHtml(selectionDescription(item))}</div>
    ${mathFormula ? `<div class="selection-formula">${mathFormula}</div>` : (formula ? `<div class="selection-body">${escapeHtml(formula)}</div>` : "")}
    <div class="selection-kv">
      <label>Source</label><span>${escapeHtml(source)}</span>
      ${dims ? `<label>Dims</label><span>${escapeHtml(dims)}</span>` : ""}
      ${params && params !== "-" ? `<label>Params</label><span>${escapeHtml(params)}</span>` : ""}
      ${item.metadata?.tooltip ? `<label>Calc</label><span>${escapeHtml(item.metadata.tooltip)}</span>` : ""}
    </div>
  `;
  renderMathIn(root);
}

function selectionDescription(item) {
  if (item.metadata?.summary) {
    return item.metadata.summary;
  }
  if (item.kind === "matrix") {
    return "Weight tensor used by an operation or function call. Dimensions are inferred from the Qwen3-VL-2B config or extracted weight metadata.";
  }
  if (item.kind === "op") {
    return "Open tensor expression op. Garnet backend handlers decide whether this lowers to TensorRT, custom CUDA, or debug/runtime logic.";
  }
  if (item.kind === "runtime") {
    return "Control-flow region from the xlang model expression, such as a loop or branch.";
  }
  if (item.kind === "block") {
    return "Model function or layer block. Click it in the graph to drill into calls, ops, matrices, and expression.";
  }
  return "High-level model graph node.";
}

function plainFormulaSummary(item) {
  const name = item.source?.function || item.label;
  const summaries = {
    VisionMLP: "Formula: y = GELU(x W_fc1^T + b_fc1) W_fc2^T + b_fc2",
    Qwen3TextMLP: "Formula: y = (SiLU(x W_gate^T) * x W_up^T) W_down^T",
    VisionAttention: "Formula: [Q,K,V] = x W_qkv^T + b; y = Attention(RoPE(Q,K), V) W_o^T",
    Qwen3TextAttention: "Formula: y = PagedAttention(RoPE(Q,K), V) W_o^T",
    linear: "Formula: y = x W^T + b"
  };
  return summaries[name] || "";
}

function repeatInfoForFunction(functionName) {
  if (functionName === "Qwen3TextDecoderLayer" || functionName === "Qwen3TextAttention" || functionName === "Qwen3TextMLP") {
    return "template, repeated 28x in text decoder";
  }
  if (functionName === "VisionBlock" || functionName === "VisionAttention" || functionName === "VisionMLP") {
    return "template, repeated 24x in vision tower";
  }
  if (functionName === "VisionPatchMerger") {
    return "used by final merger and DeepStack merger path";
  }
  return "function template";
}

function graphContextText(context) {
  if (!context) {
    return "";
  }
  const parts = [context.repeat || context.kind];
  if (context.source?.file && context.source?.line) {
    parts.push(`${context.source.file}:${context.source.line}`);
  }
  return parts.filter(Boolean).join(" | ");
}

function renderTraceRoute() {
  const root = document.getElementById("trace-table");
  const count = document.getElementById("trace-count");
  count.textContent = `${state.traceEvents.length} events`;
  if (!state.traceEvents.length) {
    root.innerHTML = `<div class="empty-state">Load a JSONL trace to inspect runtime events.</div>`;
    return;
  }

  root.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Duration</th>
          <th>Event</th>
          <th>Node</th>
          <th>Message</th>
        </tr>
      </thead>
      <tbody>
        ${state.traceEvents.map((event) => `
          <tr data-node-id="${escapeHtml(event.node_id || "")}">
            <td>${escapeHtml(String(event.ts_us ?? "-"))}</td>
            <td>${escapeHtml(String(event.duration_us ?? "-"))}</td>
            <td>${escapeHtml(event.event || "-")}</td>
            <td>${escapeHtml(event.node_id || "-")}</td>
            <td>${escapeHtml(event.message || "")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  root.querySelectorAll("tr[data-node-id]").forEach((row) => {
    row.addEventListener("click", () => {
      if (!row.dataset.nodeId) {
        return;
      }
      state.selectedId = row.dataset.nodeId;
      setRoute("detail");
      renderGraph();
      renderTable();
      renderDetail();
      renderDrillDetail();
    });
  });
}

function renderDrillDetail() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === state.detailTab);
  });

  const item = getSelectedNode();
  const title = document.getElementById("drill-title");
  const status = document.getElementById("drill-status");
  const content = document.getElementById("drill-content");

  if (!item) {
    title.textContent = "No block selected";
    status.textContent = "-";
    content.innerHTML = `<div class="empty-state">Select a graph block to inspect layers, ops, and expression.</div>`;
    return;
  }

  title.textContent = item.label;
  status.textContent = item.status || "-";

  if (state.detailTab === "layers") {
    content.innerHTML = renderLayerTab(item);
  } else if (state.detailTab === "ops") {
    content.innerHTML = renderOpsTab(item);
  } else {
    content.innerHTML = renderExpressionTab(item);
  }
  renderMath();

  content.querySelectorAll("[data-select-node]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedId = row.dataset.selectNode;
      renderGraph();
      renderTable();
      renderDetail();
      renderDrillDetail();
    });
  });
}

function renderLayerTab(item) {
  const related = uniqueNodes([...parentNodes(item.id), ...childNodes(item.id)]);
  return `
    ${renderInfoGrid(item)}
    <div class="list-card">
      ${related.length ? related.map(renderNodeRow).join("") : `<div class="empty-state">No connected layers in this graph artifact.</div>`}
    </div>
  `;
}

function renderOpsTab(item) {
  const ops = collectOpsForNode(item);
  return `
    ${renderInfoGrid(item)}
    <div class="list-card">
      ${ops.length ? ops.map(renderNodeRow).join("") : `<div class="empty-state">No op nodes connected to this block yet.</div>`}
    </div>
  `;
}

function renderExpressionTab(item) {
  const formula = buildMathFormula(item);
  return `
    ${renderInfoGrid(item)}
    ${formula ? `<div class="formula-box">${formula}</div>` : ""}
    <div class="expression-box">
      <pre>${escapeHtml(buildPseudoExpression(item))}</pre>
    </div>
  `;
}

function renderInfoGrid(item) {
  const source = item.source ? `${item.source.file}:${item.source.line}` : "-";
  const matrixMeta = item.kind === "matrix"
    ? `
      <div class="info-cell"><label>Dimensions</label><span>${escapeHtml(formatNodeDimensions(item))}</span></div>
      <div class="info-cell"><label>Role</label><span>${escapeHtml(item.metadata?.role || "-")}</span></div>
      <div class="info-cell"><label>DType</label><span>${escapeHtml(item.metadata?.dtype || "-")}</span></div>
      <div class="info-cell"><label>Params</label><span>${escapeHtml(formatMatrixParamScope(item))}</span></div>
      <div class="info-cell"><label>FP16 Size</label><span>${escapeHtml(formatBytes(item.metadata?.bytes_fp16))}</span></div>
      <div class="info-cell"><label>Dim Source</label><span>${escapeHtml(item.metadata?.dimension_source || "-")}</span></div>
    `
    : "";
  return `
    <div class="info-grid">
      <div class="info-cell"><label>ID</label><span>${escapeHtml(item.id)}</span></div>
      <div class="info-cell"><label>Kind</label><span>${escapeHtml(item.kind)}</span></div>
      <div class="info-cell"><label>Backend</label><span>${escapeHtml(item.backend || "-")}</span></div>
      <div class="info-cell"><label>Source</label><span>${escapeHtml(source)}</span></div>
      ${matrixMeta}
      ${item.metadata?.tooltip ? `<div class="info-cell"><label>Calculation</label><span>${escapeHtml(item.metadata.tooltip)}</span></div>` : ""}
    </div>
  `;
}

function renderNodeRow(item) {
  const secondary = item.kind === "matrix"
    ? formatNodeDimensions(item)
    : (item.op || item.source?.function || "-");
  return `
    <div class="list-row" data-select-node="${escapeHtml(item.id)}">
      <strong>${escapeHtml(item.label)}</strong>
      <span>${escapeHtml(secondary)}</span>
      <span>${escapeHtml(item.backend || "-")}</span>
      <span class="pill">${escapeHtml(item.status || "-")}</span>
    </div>
  `;
}

function collectOpsForNode(item) {
  if (item.kind === "op") {
    return [item];
  }
  const directOps = childNodes(item.id).filter((child) => child.kind === "op");
  if (directOps.length) {
    return directOps;
  }
  const functionName = item.source?.function;
  return graphData.nodes.filter((candidate) => candidate.kind === "op" && candidate.source?.function === functionName);
}

function childNodes(nodeId) {
  return graphData.edges.filter((edgeItem) => edgeItem.from === nodeId).map((edgeItem) => findNode(edgeItem.to)).filter(Boolean);
}

function parentNodes(nodeId) {
  return graphData.edges.filter((edgeItem) => edgeItem.to === nodeId).map((edgeItem) => findNode(edgeItem.from)).filter(Boolean);
}

function uniqueNodes(items) {
  return items.filter((item, index) => items.findIndex((candidate) => candidate.id === item.id) === index);
}

function findNode(nodeId) {
  return graphData.nodes.find((item) => item.id === nodeId);
}

function getSelectedNode() {
  return findNode(state.selectedId) || graphData.nodes[0];
}

function buildPseudoExpression(item) {
  if (item.metadata?.expression) {
    if (item.kind === "matrix") {
      return [
        item.metadata.expression,
        "",
        `# role: ${item.metadata.role || "matrix"}`,
        `# dimensions: ${formatNodeDimensions(item)}`,
        `# dtype: ${item.metadata.dtype || "unknown"}`,
        `# parameters: ${formatMatrixParamScope(item)}`,
        `# fp16_size: ${formatBytes(item.metadata.bytes_fp16)}`,
        `# dimension_source: ${item.metadata.dimension_source || "unknown"}`
      ].join("\n");
    }
    return item.metadata.expression;
  }

  const source = item.source ? `${item.source.file}:${item.source.line}` : "unknown";
  if (item.kind === "op") {
    const opCall = item.metadata?.xlang_op_kind || "binary_op/unary_op";
    return [
      `# ${item.label}`,
      `# source: ${source}`,
      `# backend: ${item.backend || "unknown"}, status: ${item.status || "unknown"}`,
      "",
      `output = input * T.${opCall}("${item.op || item.label}") * operand`,
      "",
      "# MVP note: this is graph-derived expression text.",
      "# Next step is storing actual function body snippets from the extractor."
    ].join("\n");
  }

  const ops = collectOpsForNode(item);
  const lines = [
    `# ${item.label}`,
    `# source: ${source}`,
    `# function: ${item.source?.function || "unknown"}`,
    "",
    `def ${item.source?.function || "block"}(...):`
  ];
  if (!ops.length) {
    lines.push("    # No connected op nodes in current graph JSON.");
  } else {
    ops.forEach((opNode) => {
      const opCall = opNode.metadata?.xlang_op_kind || "op";
      lines.push(`    ${slugVariable(opNode.label)} = input * T.${opCall}("${opNode.op || opNode.label}")`);
    });
  }
  lines.push("    return output");
  return lines.join("\n");
}

function buildMathFormula(item) {
  if (item.metadata?.formula) {
    return item.metadata.formula;
  }
  const name = item.source?.function || item.label;
  const label = item.label || "";
  const formulas = {
    VisionMLP: String.raw`\[
\begin{aligned}
h &= \operatorname{GELU}(x W_{fc1}^{T} + b_{fc1}) \\
y &= h W_{fc2}^{T} + b_{fc2}
\end{aligned}
\]`,
    Qwen3TextMLP: String.raw`\[
\begin{aligned}
g &= \operatorname{SiLU}(x W_{gate}^{T}) \\
u &= x W_{up}^{T} \\
y &= (g \odot u) W_{down}^{T}
\end{aligned}
\]`,
    VisionAttention: String.raw`\[
\begin{aligned}
[Q,K,V] &= x W_{qkv}^{T} + b_{qkv} \\
\hat Q,\hat K &= \operatorname{RoPE}(Q,K) \\
Y &= \operatorname{Attention}(\hat Q,\hat K,V) W_o^{T} + b_o
\end{aligned}
\]`,
    Qwen3TextAttention: String.raw`\[
\begin{aligned}
Q &= \operatorname{RMSNorm}(x W_q^{T}) \\
K &= \operatorname{RMSNorm}(x W_k^{T}) \\
V &= x W_v^{T} \\
Y &= \operatorname{PagedAttention}(\operatorname{RoPE}(Q,K), V) W_o^{T}
\end{aligned}
\]`,
    VisionPatchMerger: String.raw`\[
\begin{aligned}
s &= \operatorname{Shuffle}_{merge}(x) \\
h &= \operatorname{GELU}(\operatorname{LN}(s) W_{fc1}^{T} + b_{fc1}) \\
y &= h W_{fc2}^{T} + b_{fc2}
\end{aligned}
\]`,
    linear: String.raw`\[
y = x W^{T} + b
\]`,
    rms_norm: String.raw`\[
y = \frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}} \odot \gamma
\]`,
    layer_norm: String.raw`\[
y = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \odot \gamma + \beta
\]`
  };

  if (item.kind === "matrix") {
    return String.raw`\[
${escapeLatex(item.label)} \in \mathbb{R}^{${escapeLatex(formatNodeDimensions(item).replace(/^\[|\]$/g, ""))}}
\]`;
  }
  if (item.kind === "op") {
    return String.raw`\[
\operatorname{${escapeLatex(item.op || label)}}(\cdot)
\]`;
  }
  return formulas[name] || formulas[label] || "";
}

function escapeLatex(value) {
  return String(value || "").replace(/_/g, "\\_").replace(/#/g, "\\#");
}

function renderMath() {
  renderMathIn(document.getElementById("drill-content"));
}

function renderMathIn(root) {
  if (window.MathJax?.typesetPromise && root) {
    window.MathJax.typesetPromise([root]).catch(() => {});
  }
}

function renderRoute() {
  document.querySelectorAll(".route-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.route === state.route);
  });
  document.querySelectorAll(".route-view").forEach((view) => {
    view.classList.toggle("active", view.dataset.view === state.route);
  });
  if (state.route === "graph") {
    requestAnimationFrame(applyViewport);
  }
}

function setRoute(route, updateHash = true) {
  const nextRoute = ["graph", "detail", "ops", "trace"].includes(route) ? route : "graph";
  state.route = nextRoute;
  if (updateHash && location.hash !== `#${nextRoute}`) {
    history.replaceState(null, "", `#${nextRoute}`);
  }
  renderRoute();
}

function bindCanvasEvents() {
  const canvas = document.getElementById("graph-canvas");

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const direction = event.deltaY < 0 ? 1.12 : 0.88;
    zoomGraph(direction, event.clientX, event.clientY);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (event) => {
    if (event.target.closest("#minimap")) {
      return;
    }
    if (event.target.closest(".node-card")) {
      return;
    }
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("panning");
    state.pan = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
      originX: state.viewport.x,
      originY: state.viewport.y
    };
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.pan.active) {
      return;
    }
    state.viewport.x = state.pan.originX + event.clientX - state.pan.startX;
    state.viewport.y = state.pan.originY + event.clientY - state.pan.startY;
    applyViewport();
  });

  canvas.addEventListener("pointerup", (event) => {
    state.pan.active = false;
    canvas.classList.remove("panning");
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  });

  canvas.addEventListener("pointercancel", () => {
    state.pan.active = false;
    canvas.classList.remove("panning");
  });
}

function bindMinimapEvents() {
  const canvas = document.getElementById("graph-canvas");
  canvas.addEventListener("pointerdown", (event) => {
    const minimap = event.target.closest("#minimap");
    if (!minimap) {
      return;
    }
    event.stopPropagation();
    canvas.setPointerCapture(event.pointerId);
    jumpViewportFromMinimap(event, minimap);
    const move = (moveEvent) => jumpViewportFromMinimap(moveEvent, minimap);
    const up = (upEvent) => {
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      if (canvas.hasPointerCapture(upEvent.pointerId)) {
        canvas.releasePointerCapture(upEvent.pointerId);
      }
    };
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
  });
}

function bindSplitters() {
  const appSplitter = document.getElementById("app-splitter");
  const hSplitter = document.getElementById("sidebar-hsplitter");

  appSplitter?.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    appSplitter.setPointerCapture(event.pointerId);
    appSplitter.classList.add("dragging");
    document.body.classList.add("resizing");

    const move = (moveEvent) => {
      const width = clamp(moveEvent.clientX, 220, 520);
      document.documentElement.style.setProperty("--sidebar-width", `${width}px`);
      requestAnimationFrame(applyViewport);
    };
    const up = (upEvent) => {
      appSplitter.classList.remove("dragging");
      document.body.classList.remove("resizing");
      appSplitter.removeEventListener("pointermove", move);
      appSplitter.removeEventListener("pointerup", up);
      if (appSplitter.hasPointerCapture(upEvent.pointerId)) {
        appSplitter.releasePointerCapture(upEvent.pointerId);
      }
    };

    appSplitter.addEventListener("pointermove", move);
    appSplitter.addEventListener("pointerup", up);
  });

  hSplitter?.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    hSplitter.setPointerCapture(event.pointerId);
    hSplitter.classList.add("dragging");
    document.body.classList.add("resizing-y");
    const sidebarRect = document.querySelector(".sidebar").getBoundingClientRect();

    const move = (moveEvent) => {
      const maxHeight = Math.max(180, sidebarRect.height - 260);
      const height = clamp(sidebarRect.bottom - moveEvent.clientY, 140, maxHeight);
      document.documentElement.style.setProperty("--inspector-height", `${height}px`);
    };
    const up = (upEvent) => {
      hSplitter.classList.remove("dragging");
      document.body.classList.remove("resizing-y");
      hSplitter.removeEventListener("pointermove", move);
      hSplitter.removeEventListener("pointerup", up);
      if (hSplitter.hasPointerCapture(upEvent.pointerId)) {
        hSplitter.releasePointerCapture(upEvent.pointerId);
      }
    };

    hSplitter.addEventListener("pointermove", move);
    hSplitter.addEventListener("pointerup", up);
  });
}

function zoomGraph(factor, clientX, clientY) {
  const canvas = document.getElementById("graph-canvas");
  const rect = canvas.getBoundingClientRect();
  const anchorX = clientX ?? (rect.left + rect.width / 2);
  const anchorY = clientY ?? (rect.top + rect.height / 2);
  const localX = anchorX - rect.left;
  const localY = anchorY - rect.top;
  const oldScale = state.viewport.scale;
  const nextScale = clamp(oldScale * factor, 0.25, 3.5);
  const graphX = (localX - state.viewport.x) / oldScale;
  const graphY = (localY - state.viewport.y) / oldScale;
  state.viewport.scale = nextScale;
  state.viewport.x = localX - graphX * nextScale;
  state.viewport.y = localY - graphY * nextScale;
  applyViewport();
}

function fitGraph() {
  const canvas = document.getElementById("graph-canvas");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }
  const scale = clamp(Math.min(rect.width / state.graphSize.width, rect.height / state.graphSize.height) * 0.92, 0.25, 2);
  state.viewport.scale = scale;
  state.viewport.x = (rect.width - state.graphSize.width * scale) / 2;
  state.viewport.y = (rect.height - state.graphSize.height * scale) / 2;
  applyViewport();
}

function applyViewport() {
  const viewport = document.getElementById("graph-viewport");
  const label = document.getElementById("zoom-label");
  if (viewport) {
    viewport.setAttribute("transform", `translate(${state.viewport.x} ${state.viewport.y}) scale(${state.viewport.scale})`);
  }
  if (label) {
    label.textContent = `${Math.round(state.viewport.scale * 100)}%`;
  }
  updateMinimapViewport();
}

function renderMinimap() {
  const minimap = document.getElementById("minimap");
  if (!minimap || !state.lastGraphView) {
    return;
  }
  const nodes = state.lastGraphView.nodes.map((item) => {
    const pos = state.lastPositions[item.id];
    if (!pos) {
      return "";
    }
    return `<rect class="minimap-node ${escapeHtml(item.kind)}" x="${pos.x}" y="${pos.y}" width="${CARD_WIDTH}" height="${CARD_HEIGHT}"></rect>`;
  }).join("");
  minimap.innerHTML = `
    <svg viewBox="0 0 ${state.graphSize.width} ${state.graphSize.height}" preserveAspectRatio="xMidYMid meet">
      ${nodes}
      <rect id="minimap-view" class="minimap-view" x="0" y="0" width="0" height="0"></rect>
    </svg>
  `;
  updateMinimapViewport();
}

function updateMinimapViewport() {
  const rect = document.getElementById("minimap-view");
  const canvas = document.getElementById("graph-canvas");
  if (!rect || !canvas) {
    return;
  }
  const bounds = canvas.getBoundingClientRect();
  rect.setAttribute("x", String(-state.viewport.x / state.viewport.scale));
  rect.setAttribute("y", String(-state.viewport.y / state.viewport.scale));
  rect.setAttribute("width", String(bounds.width / state.viewport.scale));
  rect.setAttribute("height", String(bounds.height / state.viewport.scale));
}

function jumpViewportFromMinimap(event, minimap) {
  const svg = minimap.querySelector("svg");
  const canvas = document.getElementById("graph-canvas");
  if (!svg || !canvas) {
    return;
  }
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const matrix = svg.getScreenCTM();
  if (!matrix) {
    return;
  }
  const graphPoint = point.matrixTransform(matrix.inverse());
  const canvasRect = canvas.getBoundingClientRect();
  state.viewport.x = canvasRect.width / 2 - graphPoint.x * state.viewport.scale;
  state.viewport.y = canvasRect.height / 2 - graphPoint.y * state.viewport.scale;
  applyViewport();
}

async function handleTraceFile(event) {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  state.traceEvents = parseTrace(text);
  renderDetail();
  event.target.value = "";
}

async function handleGraphFile(event) {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  try {
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
      throw new Error("Graph JSON must contain nodes and edges arrays.");
    }
    graphData = parsed;
    state.selectedId = graphData.nodes[0]?.id || "";
    state.query = "";
    state.traceEvents = [];
    state.enabledStatuses = new Set(graphData.nodes.map((item) => item.status));
    state.viewport = { x: 20, y: 20, scale: 1 };
    document.getElementById("search-input").value = "";
    render();
  } catch (error) {
    alert(`Could not load graph JSON: ${error.message}`);
  } finally {
    event.target.value = "";
  }
}

function parseTrace(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }
  if (trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return trimmed.split(/\r?\n/).map((line) => {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }).filter(Boolean);
}

function shorten(value, maxLength) {
  const text = String(value || "");
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1)}...`;
}

function matrixRepeatCount(item) {
  const functionName = item.source?.function || "";
  const config = graphData.model?.config || {};
  if (/^Qwen3Text/.test(functionName)) {
    return config.text_layers || 28;
  }
  if (/^(VisionBlock|VisionAttention|VisionMLP)$/.test(functionName)) {
    return config.vision_depth || 24;
  }
  return 1;
}

function formatMatrixParamScope(item) {
  const params = item.metadata?.parameter_count;
  if (!Number.isFinite(params)) {
    return "-";
  }
  const repeat = matrixRepeatCount(item);
  if (repeat > 1) {
    return `${formatParamCount(params)}/layer x${repeat} = ${formatParamCount(params * repeat)}`;
  }
  return `${formatParamCount(params)} template`;
}

function formatMatrixParamCard(item) {
  const params = item.metadata?.parameter_count;
  if (!Number.isFinite(params)) {
    return "-";
  }
  const repeat = matrixRepeatCount(item);
  if (repeat > 1) {
    return `${formatParamCount(params)}/layer x${repeat}`;
  }
  return `${formatParamCount(params)} template`;
}

function nodeSubtitle(item) {
  if (item.metadata?.role === "formula" && item.metadata?.dims) {
    return `${item.op || item.kind} | ${item.metadata.dims}`;
  }
  if (item.metadata?.summary) {
    return item.metadata.summary;
  }
  if (item.kind === "matrix") {
    const parts = [formatNodeDimensions(item)];
    const params = formatMatrixParamCard(item);
    if (params !== "-") {
      parts.push(params);
    }
    if (item.metadata?.dtype) {
      parts.push(item.metadata.dtype);
    }
    return parts.join(" | ");
  }
  return item.op || item.kind;
}

function nodeTooltip(item) {
  if (item.metadata?.tooltip) {
    return item.metadata.tooltip;
  }
  if (item.kind === "matrix") {
    const repeat = matrixRepeatCount(item);
    const params = item.metadata?.parameter_count;
    const scopedParams = formatMatrixParamScope(item);
    return [
      item.label,
      `dimensions: ${formatNodeDimensions(item)}`,
      `params: ${scopedParams}`,
      repeat > 1 && Number.isFinite(params) ? `meaning: ${formatParamCount(params)} is one layer/template copy, not the full model` : "",
      `fp16/bf16 bytes: ${formatBytes(item.metadata?.bytes_fp16)}`,
      `source: ${item.metadata?.dimension_source || "unknown"}`
    ].filter(Boolean).join("\n");
  }
  return `${item.label}${item.op ? `\n${item.op}` : ""}`;
}

function formatNodeDimensions(item) {
  const dims = item.metadata?.dimensions;
  if (Array.isArray(dims) && dims.length) {
    return `[${dims.join(", ")}]`;
  }
  const outputs = item.shape?.outputs;
  if (Array.isArray(outputs) && outputs.length) {
    return outputs.join(", ");
  }
  return "dim: unknown";
}

function formatParamCount(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return String(value);
}

function formatBytes(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (value >= 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  }
  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(2)} MiB`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)} KiB`;
  }
  return `${value} B`;
}

function svgTextLines(value, x, y, lineHeight, maxChars, className, maxLines) {
  const words = String(value || "").split(/([._/:-])/).flatMap((part) => part.includes(" ") ? part.split(/\s+/) : [part]).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const needsSpace = line && !/^[._/:-]$/.test(word) && !/[._/:-]$/.test(line);
    const next = `${line}${needsSpace ? " " : ""}${word}`;
    if (next.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  });
  if (line) {
    lines.push(line);
  }
  const limited = lines.slice(0, maxLines);
  return `
    <text ${className ? `class="${className}"` : ""} x="${x}" y="${y}">
      ${limited.map((text, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeSvg(text)}</tspan>`).join("")}
    </text>
  `;
}

function edgeLabelSvg(value, x1, y1, x2, y2, dimmed) {
  const text = String(value || "");
  const x = (x1 + x2) / 2;
  const y = (y1 + y2) / 2 - 8;
  return `
    <g class="edge-label" opacity="${dimmed ? "0.18" : "1"}">
      <text x="${x}" y="${y + 2}">${escapeSvg(text)}</text>
    </g>
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeSvg(value) {
  return escapeHtml(value);
}

function slugVariable(value) {
  return String(value || "op").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "op";
}

function titleCase(value) {
  return String(value || "").replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

setRoute(location.hash.replace("#", "") || "graph", false);
init();
