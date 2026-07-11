import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE05 = REPO_ROOT / "test2026" / "tests" / "phase_05_subgraph_parity" / "test_real_qwen_mlp_subgraphs.py"
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_text_cached_prefill_decode"


def main():
    env = os.environ.copy()
    env.setdefault("RUN_GARNET_REAL_QWEN_MLP_PARITY", "1")
    env["RUN_GARNET_REAL_QWEN_CACHED_PREFILL_DECODE_PARITY"] = "1"
    env["GARNET_CACHED_PREFILL_DECODE_ONLY"] = "1"
    env.setdefault("GARNET_CACHED_DECODE_PROMPT_TOKENS", "11")
    env.setdefault("GARNET_CACHED_DECODE_PAGE_SIZE", "8")
    env.setdefault("GARNET_CACHED_DECODE_LAYER_COUNT", "2")

    completed = subprocess.run(
        [sys.executable, "-u", str(PHASE05)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "phase_17_output.log").write_text(completed.stdout, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        raise AssertionError(f"Phase 17 cached prefill/decode parity failed rc={completed.returncode}\n{completed.stdout}")

    summary_src = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs" / "real_qwen_cached_prefill_decode_parity.json"
    if not summary_src.exists():
        raise AssertionError(f"Phase 17 summary was not produced. Output:\n{completed.stdout}")
    summary = json.loads(summary_src.read_text(encoding="utf-8"))
    result = summary["cached_prefill_decode"]
    (ARTIFACT_DIR / "real_qwen_cached_prefill_decode_parity.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Phase 17: Qwen text cached prefill/decode parity passed.")
    print(
        f"layers={result['layer_count']} prompt_tokens={result['prompt_tokens']} "
        f"page_size={result['page_size']} pages={result['logical_pages']}/{result['physical_pages']}"
    )
    print(f"hidden_max_error={result['hidden_max_error']:.8f} hidden_mean_error={result['hidden_mean_error']:.8f}")
    print(f"logits_max_error={result['logits_max_error']:.8f} logits_mean_error={result['logits_mean_error']:.8f}")
    print(f"top1_match={result['top1_match']} expected_top1={result['expected_top1']} actual_top1={result['actual_top1']}")
    print(f"wrote {ARTIFACT_DIR / 'real_qwen_cached_prefill_decode_parity.json'}")


if __name__ == "__main__":
    main()
