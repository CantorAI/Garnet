// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";
import { app, xlang } from "electron/main";

async function callJson(module, name, args = []) {
  return JSON.parse(await module.call(name, args));
}

async function main() {
  await app.whenReady();
  const runtimeRoot = path.resolve(process.env.GARNET_RUNTIME_ROOT ?? process.cwd());
  await xlang.initialize({
    libraryPath: runtimeRoot,
    appPath: runtimeRoot,
    librarySearchPaths: [runtimeRoot],
    enablePython: false,
  });

  const garnet = await xlang.importModule("garnet", { fromPath: runtimeRoot });
  let serving = false;
  try {
    if (process.env.GARNET_ACCELERATION_ROOT) {
      const activated = await callJson(garnet, "activate_acceleration_path_json", [
        path.resolve(process.env.GARNET_ACCELERATION_ROOT),
      ]);
      if (activated.status !== "ready") throw new Error(JSON.stringify(activated));
    }
    console.log(await callJson(garnet, "detect_acceleration_json"));

    if (!process.env.GARNET_MODEL_ROOT) {
      console.log("Set GARNET_MODEL_ROOT to run real Qwen3-1.7B inference.");
      return;
    }
    const modelRoot = path.resolve(process.env.GARNET_MODEL_ROOT);
    const xmodelRoot = path.resolve(process.env.GARNET_XMODEL_ROOT ?? path.join(modelRoot, "xmodel"));
    const cacheRoot = path.resolve(process.env.GARNET_CACHE_ROOT ?? path.join(modelRoot, "compiled_cache"));
    const loaded = await callJson(garnet, "serve_model", [
      modelRoot, xmodelRoot, cacheRoot, "{}", "Qwen3-1.7B",
    ]);
    if (!loaded.ready) throw new Error(JSON.stringify(loaded));
    serving = true;
    const result = await callJson(garnet, "infer_json", [
      process.env.GARNET_PROMPT ?? "Explain Garnet in one sentence.", "", 64,
    ]);
    if (result.status !== "ok") throw new Error(JSON.stringify(result));
    console.log(result.text);
  } finally {
    if (serving) await garnet.call("stop_serving", []);
    if (garnet.dispose) await garnet.dispose();
    await xlang.shutdown();
  }
}

main()
  .then(() => app.exit(0))
  .catch((error) => {
    console.error(error);
    app.exit(1);
  });
