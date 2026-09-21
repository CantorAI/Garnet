# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SCREENSHOT_DIR = REPO_ROOT / "test2026" / "fixtures" / "vlm_screenshots"
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "codex_screen_code_generation"
CODEX = shutil.which("codex")

assert CODEX, "codex CLI is not available on PATH"
screenshots = sorted(SCREENSHOT_DIR.glob("*.jpg"))
assert len(screenshots) == 2, f"expected two JPG screenshots in {SCREENSHOT_DIR}, got {screenshots}"

expected_methods = {
    "Q1.jpg": "minimizeResult",
    "Q2.jpg": "getResults",
}
prompt = (
    "Examine the attached screenshot and extract the coding question yourself, "
    "including its required interface, constraints, and examples. Then solve the "
    "extracted question. Return only one complete, correct Python 3 submission with "
    "all required imports and the exact required interface. Output raw Python code "
    "only: no markdown fences or explanation. Do not modify files."
)


def remove_markdown_fence(text):
    code = text.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].lstrip()
    elif code.startswith("```"):
        code = code[3:].lstrip()
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code


def validate_structure(code, expected_method):
    tree = ast.parse(code)
    solution_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Solution"
    ]
    assert len(solution_classes) == 1, "expected exactly one class Solution"
    methods = [
        node.name for node in solution_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert expected_method in methods, f"missing Solution.{expected_method}"


def functional_validator(image_name):
    if image_name == "Q1.jpg":
        return r'''
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
solution = namespace["Solution"]()

def minimum_answers(expression):
    left, right = expression.split("+")
    best = None
    answers = set()
    for left_index in range(len(left)):
        for right_index in range(1, len(right) + 1):
            left_factor = int(left[:left_index]) if left_index else 1
            middle = int(left[left_index:]) + int(right[:right_index])
            right_factor = int(right[right_index:]) if right_index < len(right) else 1
            value = left_factor * middle * right_factor
            candidate = (
                left[:left_index]
                + "("
                + left[left_index:]
                + "+"
                + right[:right_index]
                + ")"
                + right[right_index:]
            )
            if best is None or value < best:
                best = value
                answers = {candidate}
            elif value == best:
                answers.add(candidate)
    return answers

for expression in ["247+38", "12+34", "999+999", "1+1", "123+456", "90+09"]:
    actual = solution.minimizeResult(expression)
    expected = minimum_answers(expression)
    assert actual in expected, (expression, actual, sorted(expected))
'''
    if image_name == "Q2.jpg":
        return r'''
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
solution = namespace["Solution"]()

cases = [
    (
        [[1, 2], [2, 3, 3], [2, 3, 1], [2, 2, 2]],
        [False, True, True],
    ),
    (
        [[1, 5], [1, 10], [2, 12, 5], [2, 12, 6], [2, 5, 5], [2, 4, 5]],
        [True, False, True, False],
    ),
    (
        [[2, 10, 10], [1, 4], [2, 10, 6], [2, 10, 4]],
        [True, True, True],
    ),
]
for queries, expected in cases:
    actual = solution.getResults([query[:] for query in queries])
    assert actual == expected, (queries, actual, expected)
'''
    raise AssertionError(f"no validator for {image_name}")


def run_functional_validation(solution_path, image_name):
    completed = subprocess.run(
        [sys.executable, "-c", functional_validator(image_name), str(solution_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
results = []
for screenshot in screenshots:
    stem = screenshot.stem.lower()
    raw_output_path = ARTIFACT_DIR / f"{stem}_raw.txt"
    solution_path = ARTIFACT_DIR / f"{stem}_solution.py"
    command = [
        CODEX,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--image",
        str(screenshot),
        "--output-last-message",
        str(raw_output_path),
        "-C",
        str(REPO_ROOT),
        prompt,
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert completed.returncode == 0, (
        screenshot.name,
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
    assert raw_output_path.exists(), raw_output_path

    raw_text = raw_output_path.read_text(encoding="utf-8")
    code = remove_markdown_fence(raw_text)
    expected_method = expected_methods[screenshot.name]
    structure_error = ""
    try:
        validate_structure(code, expected_method)
        structure_valid = True
    except (AssertionError, SyntaxError) as error:
        structure_valid = False
        structure_error = str(error)

    functional = {
        "passed": False,
        "return_code": None,
        "stdout": "",
        "stderr": "structure validation failed",
    }
    if structure_valid:
        solution_path.write_text(code + "\n", encoding="utf-8")
        functional = run_functional_validation(solution_path, screenshot.name)

    results.append({
        "image": screenshot.name,
        "elapsed_ms": elapsed_ms,
        "expected_method": expected_method,
        "structure_valid": structure_valid,
        "structure_error": structure_error,
        "functional_validation": functional,
        "raw_output_path": str(raw_output_path),
        "solution_path": str(solution_path) if solution_path.exists() else "",
        "code": code,
        "codex_stdout": completed.stdout,
        "codex_stderr": completed.stderr,
    })

artifact_path = ARTIFACT_DIR / "results.json"
artifact_path.write_text(
    json.dumps(
        {
            "runner": "codex exec",
            "codex_path": CODEX,
            "prompt": prompt,
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

for item in results:
    print(
        f"{item['image']}: elapsed_ms={item['elapsed_ms']:.2f}, "
        f"method={item['expected_method']}, structure_valid={item['structure_valid']}, "
        f"functional_valid={item['functional_validation']['passed']}, "
        f"solution={item['solution_path']}"
    )
assert all(
    item["structure_valid"] and item["functional_validation"]["passed"]
    for item in results
), f"Codex-generated solutions failed validation; inspect {artifact_path}"
print(f"Codex screenshot code generation passed: artifact={artifact_path}")
