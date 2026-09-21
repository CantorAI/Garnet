# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE05 = REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "test_real_qwen_mlp_subgraphs.py"
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_text_cached_decoder_layer"


def main():
    env = os.environ.copy()
    env.setdefault("RUN_GARNET_REAL_QWEN_MLP_PARITY", "1")
    env["RUN_GARNET_REAL_QWEN_CACHED_DECODER_LAYER_PARITY"] = "1"
    env["GARNET_CACHED_DECODER_LAYER_ONLY"] = "1"
    env.setdefault("GARNET_CACHED_DECODER_LAYER_TOKENS", "17")
    env.setdefault("GARNET_CACHED_DECODER_LAYER_PAGE_SIZE", "8")

    command = [sys.executable, "-u", str(PHASE05)]
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_DIR / "phase_16_output.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    summary_src = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs" / "real_qwen_cached_decoder_layer_parity.json"
    summary = None
    if summary_src.exists():
        summary = json.loads(summary_src.read_text(encoding="utf-8"))
        (ARTIFACT_DIR / "real_qwen_cached_decoder_layer_parity.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    if proc.returncode != 0:
        raise AssertionError(f"Phase 16 cached decoder layer parity failed rc={proc.returncode}\n{proc.stdout}")

    if summary is None or "cached_decoder_layer" not in summary:
        raise AssertionError(f"Phase 16 did not produce cached decoder summary. Output:\n{proc.stdout}")

    result = summary["cached_decoder_layer"]
    print("Phase 16: Qwen text cached decoder layer parity passed.")
    print(f"tokens={result['tokens']} page_size={result['page_size']} pages={result['logical_pages']}/{result['physical_pages']}")
    print(f"max_error={result['max_error']:.8f} mean_error={result['mean_error']:.8f}")
    print(f"wrote {ARTIFACT_DIR / 'real_qwen_cached_decoder_layer_parity.json'}")


if __name__ == "__main__":
    main()
