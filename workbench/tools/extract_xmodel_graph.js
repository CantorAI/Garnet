#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..", "..");
const defaultInputs = [
  "xModel/qwen3/vl_2b_instruct/qwen_vl_model.x",
  "xModel/qwen3/vl_2b_instruct/vision_encoder.x",
  "xModel/qwen3/vl_2b_instruct/vl_adapter.x",
  "xModel/qwen3/vl_2b_instruct/qwen_llm.x"
];

const qwen3Vl2B = {
  vocab_size: 151936,
  text_hidden_size: 2048,
  text_intermediate_size: 6144,
  vision_hidden_size: 1024,
  vision_intermediate_size: 4096,
  vision_merger_hidden_size: 4096,
  vision_num_position_embeddings: 2304,
  vision_in_channels: 3,
  vision_temporal_patch_size: 2,
  vision_patch_size: 16
};

const args = process.argv.slice(2);
const outIndex = args.indexOf("--out");
const outPath = outIndex >= 0 ? args[outIndex + 1] : "";
const jsOutIndex = args.indexOf("--js-out");
const jsOutPath = jsOutIndex >= 0 ? args[jsOutIndex + 1] : "";
const inputFiles = args.filter((arg, index) => {
  if (arg === "--out" || index === outIndex + 1 || arg === "--js-out" || index === jsOutIndex + 1) {
    return false;
  }
  return !arg.startsWith("--");
});

const files = inputFiles.length ? inputFiles : defaultInputs;
const graph = extractGraph(files);
const json = `${JSON.stringify(graph, null, 2)}\n`;

if (outPath) {
  fs.mkdirSync(path.dirname(path.resolve(repoRoot, outPath)), { recursive: true });
  fs.writeFileSync(path.resolve(repoRoot, outPath), json, "utf8");
}

if (jsOutPath) {
  const js = `window.GARNET_DEFAULT_GRAPH = ${JSON.stringify(graph, null, 2)};\n`;
  fs.mkdirSync(path.dirname(path.resolve(repoRoot, jsOutPath)), { recursive: true });
  fs.writeFileSync(path.resolve(repoRoot, jsOutPath), js, "utf8");
}

if (!outPath && !jsOutPath) {
  process.stdout.write(json);
}

function extractGraph(filesToRead) {
  const nodes = [];
  const edges = [];
  const seen = new Set();
  let previousBlock = "";

  for (const relativeFile of filesToRead) {
    const absoluteFile = path.resolve(repoRoot, relativeFile);
    const source = fs.readFileSync(absoluteFile, "utf8");
    const lines = source.split(/\r?\n/);
    let currentFunction = "module";
    let currentBlock = "";

    lines.forEach((line, index) => {
      const lineNumber = index + 1;
      const functionMatch = line.match(/^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
      if (functionMatch) {
        currentFunction = functionMatch[1];
        const functionBody = readFunctionBody(lines, index);
        currentBlock = addNode(nodes, seen, {
          id: `block.${currentFunction}`,
          kind: "block",
          label: currentFunction,
          status: "defined",
          source: {
            file: normalizePath(relativeFile),
            function: currentFunction,
            line: lineNumber
          },
          metadata: {
            extractor: "static_xmodel",
            expression: functionBody
          }
        });

        if (previousBlock && previousBlock !== currentBlock) {
          edges.push({ from: previousBlock, to: currentBlock, label: "next block" });
        }
        previousBlock = currentBlock;
      }

      const opMatches = extractOpMatches(lines, index);
      const opIds = [];
      opMatches.forEach((match) => {
        const opKind = match[1];
        const opName = match[2];
        const opId = addNode(nodes, seen, {
          id: `op.${slug(currentFunction)}.${slug(opName)}.${lineNumber}`,
          kind: "op",
          label: opName,
          op: opName,
          backend: inferBackend(opName),
          status: "defined",
          source: {
            file: normalizePath(relativeFile),
            function: currentFunction,
            line: lineNumber
          },
          metadata: {
            xlang_op_kind: opKind,
            extractor: "static_xmodel",
            expression: line.trim()
          }
        });

        if (currentBlock) {
          edges.push({ from: currentBlock, to: opId, label: opKind });
        }
        opIds.push(opId);
      });

      const weightNames = extractWeightNames(line, currentFunction);
      weightNames.forEach((weightName) => {
        const weightMeta = inferWeightMetadata(weightName);
        const matrixId = addNode(nodes, seen, {
          id: `matrix.${slug(weightName)}`,
          kind: "matrix",
          label: weightName,
          status: "defined",
          shape: {
            outputs: weightMeta.dimensions ? [formatDimensions(weightMeta.dimensions)] : []
          },
          source: {
            file: normalizePath(relativeFile),
            function: currentFunction,
            line: lineNumber
          },
          metadata: {
            extractor: "static_xmodel",
            role: weightMeta.role,
            dimensions: weightMeta.dimensions,
            dtype: weightMeta.dtype,
            parameter_count: weightMeta.parameter_count,
            bytes_fp16: weightMeta.bytes_fp16,
            dimension_source: weightMeta.dimension_source,
            expression: `weights["${weightName}"]`
          }
        });

        if (opIds.length) {
          opIds.forEach((opId) => edges.push({ from: matrixId, to: opId, label: "weight" }));
        } else if (currentBlock) {
          edges.push({ from: matrixId, to: currentBlock, label: "weight" });
        }
      });
    });
  }

  return {
    schema_version: "0.1",
    model: {
      name: "Qwen3-VL from xmodel",
      family: "qwen3_vl",
      source: "xModel/qwen3/vl_2b_instruct/qwen_vl_model.x"
    },
    nodes,
    edges: dedupeEdges(edges)
  };
}

function extractWeightNames(line, currentFunction) {
  const names = [];
  for (const match of line.matchAll(/weights\[\s*["']([^"']+)["']\s*\]/g)) {
    names.push(match[1]);
  }
  for (const match of line.matchAll(/weights\[\s*prefix\s*\+\s*["']([^"']+)["']\s*\]/g)) {
    names.push(`${currentFunction}${match[1]}`);
  }
  return [...new Set(names)];
}

function extractOpMatches(lines, index) {
  const line = lines[index];
  const direct = [...line.matchAll(/T\.(binary_op|unary_op)\(\s*["']([^"']+)["']/g)];
  if (direct.length) {
    return direct;
  }

  const start = line.match(/T\.(binary_op|unary_op)\(\s*$/);
  if (!start) {
    return [];
  }

  const windowText = lines.slice(index, Math.min(lines.length, index + 6)).join(" ");
  const name = windowText.match(/T\.(binary_op|unary_op)\(\s*["']([^"']+)["']/);
  if (!name) {
    return [];
  }
  return [{
    1: name[1],
    2: name[2]
  }];
}

function addNode(nodes, seen, node) {
  if (!seen.has(node.id)) {
    nodes.push(node);
    seen.add(node.id);
  }
  return node.id;
}

function inferBackend(opName) {
  if (opName.includes("paged") || opName.includes("rope") || opName.includes("mrope")) {
    return "cuda";
  }
  if (opName.includes("attention")) {
    return "cuda";
  }
  if (opName.includes("embedding") || opName.includes("merge_visual")) {
    return "cuda";
  }
  if (opName.includes("linear") || opName.includes("lm_head") || opName.includes("conv3d")) {
    return "trt";
  }
  if (opName.includes("norm") || opName.includes("gelu") || opName.includes("silu")) {
    return "cuda";
  }
  return "unknown";
}

function inferWeightRole(weightName) {
  if (weightName.includes("embed_tokens") || weightName.includes("lm_head")) {
    return "embedding_matrix";
  }
  if (weightName.includes("q_proj") || weightName.includes("k_proj") || weightName.includes("v_proj") || weightName.includes("o_proj") || weightName.includes("qkv")) {
    return "attention_matrix";
  }
  if (weightName.includes("gate_proj") || weightName.includes("up_proj") || weightName.includes("down_proj") || weightName.includes("linear_fc")) {
    return "mlp_matrix";
  }
  if (weightName.includes("norm")) {
    return "normalization_vector";
  }
  if (weightName.includes("patch_embed")) {
    return "patch_embed_kernel";
  }
  return "matrix";
}

function inferWeightMetadata(weightName) {
  const dimensions = inferWeightDimensions(weightName);
  return {
    role: inferWeightRole(weightName),
    dimensions,
    dtype: "fp16/bf16",
    parameter_count: Array.isArray(dimensions) && dimensions.every((value) => Number.isFinite(value))
      ? dimensions.reduce((product, value) => product * value, 1)
      : null,
    bytes_fp16: Array.isArray(dimensions) && dimensions.every((value) => Number.isFinite(value))
      ? dimensions.reduce((product, value) => product * value, 1) * 2
      : null,
    dimension_source: dimensions ? "qwen3_vl_2b_config_inferred" : "unknown"
  };
}

function inferWeightDimensions(weightName) {
  if (weightName === "language_model.embed_tokens.weight") {
    return [qwen3Vl2B.vocab_size, qwen3Vl2B.text_hidden_size];
  }
  if (weightName === "language_model.norm.weight") {
    return [qwen3Vl2B.text_hidden_size];
  }
  if (weightName === "visual.patch_embed.proj.weight") {
    return [
      qwen3Vl2B.vision_hidden_size,
      qwen3Vl2B.vision_in_channels,
      qwen3Vl2B.vision_temporal_patch_size,
      qwen3Vl2B.vision_patch_size,
      qwen3Vl2B.vision_patch_size
    ];
  }
  if (weightName === "visual.patch_embed.proj.bias") {
    return [qwen3Vl2B.vision_hidden_size];
  }
  if (weightName === "visual.pos_embed.weight") {
    return [qwen3Vl2B.vision_num_position_embeddings, qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.endsWith(".q_proj.weight")) {
    return [qwen3Vl2B.text_hidden_size, qwen3Vl2B.text_hidden_size];
  }
  if (weightName.endsWith(".k_proj.weight") || weightName.endsWith(".v_proj.weight")) {
    return [1024, qwen3Vl2B.text_hidden_size];
  }
  if (weightName.endsWith(".o_proj.weight")) {
    return [qwen3Vl2B.text_hidden_size, qwen3Vl2B.text_hidden_size];
  }
  if (weightName.endsWith(".gate_proj.weight") || weightName.endsWith(".up_proj.weight")) {
    return [qwen3Vl2B.text_intermediate_size, qwen3Vl2B.text_hidden_size];
  }
  if (weightName.endsWith(".down_proj.weight")) {
    return [qwen3Vl2B.text_hidden_size, qwen3Vl2B.text_intermediate_size];
  }
  if (weightName.endsWith(".q_norm.weight") || weightName.endsWith(".k_norm.weight")) {
    return [128];
  }
  if (weightName.endsWith(".input_layernorm.weight") || weightName.endsWith(".post_attention_layernorm.weight")) {
    return [qwen3Vl2B.text_hidden_size];
  }
  if (weightName.includes("VisionMLP.linear_fc1.weight")) {
    return [qwen3Vl2B.vision_intermediate_size, qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.includes("VisionMLP.linear_fc1.bias")) {
    return [qwen3Vl2B.vision_intermediate_size];
  }
  if (weightName.includes("VisionMLP.linear_fc2.weight")) {
    return [qwen3Vl2B.vision_hidden_size, qwen3Vl2B.vision_intermediate_size];
  }
  if (weightName.includes("VisionMLP.linear_fc2.bias")) {
    return [qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.includes("VisionPatchMerger.norm.weight") || weightName.includes("VisionPatchMerger.norm.bias")) {
    return [qwen3Vl2B.vision_merger_hidden_size];
  }
  if (weightName.includes("VisionPatchMerger.linear_fc1.weight")) {
    return [qwen3Vl2B.vision_merger_hidden_size, qwen3Vl2B.vision_merger_hidden_size];
  }
  if (weightName.includes("VisionPatchMerger.linear_fc1.bias")) {
    return [qwen3Vl2B.vision_merger_hidden_size];
  }
  if (weightName.includes("VisionPatchMerger.linear_fc2.weight")) {
    return [qwen3Vl2B.text_hidden_size, qwen3Vl2B.vision_merger_hidden_size];
  }
  if (weightName.includes("VisionPatchMerger.linear_fc2.bias")) {
    return [qwen3Vl2B.text_hidden_size];
  }
  if (weightName.includes("VisionAttention.qkv.weight")) {
    return [qwen3Vl2B.vision_hidden_size * 3, qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.includes("VisionAttention.qkv.bias")) {
    return [qwen3Vl2B.vision_hidden_size * 3];
  }
  if (weightName.includes("VisionAttention.proj.weight")) {
    return [qwen3Vl2B.vision_hidden_size, qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.includes("VisionAttention.proj.bias")) {
    return [qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.includes("VisionBlock.norm1") || weightName.includes("VisionBlock.norm2")) {
    return [qwen3Vl2B.vision_hidden_size];
  }
  if (weightName.endsWith(".norm.weight") || weightName.endsWith(".norm.bias")) {
    return ["hidden"];
  }
  if (weightName.endsWith(".bias")) {
    return ["out_features"];
  }
  if (weightName.endsWith(".weight")) {
    return ["out_features", "in_features"];
  }
  return null;
}

function formatDimensions(dimensions) {
  return `[${dimensions.join(", ")}]`;
}

function dedupeEdges(edges) {
  const seen = new Set();
  return edges.filter((edge) => {
    const key = `${edge.from}->${edge.to}:${edge.label || ""}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function slug(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
}

function normalizePath(value) {
  return value.replace(/\\/g, "/");
}

function readFunctionBody(lines, startIndex) {
  const body = [];
  for (let i = startIndex; i < lines.length; i += 1) {
    const line = lines[i];
    if (i !== startIndex && /^\S/.test(line) && line.trim()) {
      break;
    }
    body.push(line);
    if (body.length >= 80) {
      body.push("    # ... truncated by Workbench extractor ...");
      break;
    }
  }
  return body.join("\n");
}
