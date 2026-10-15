"""Index existing experiment reports without rewriting or reinterpreting evidence.

Run with Python from any directory. Generated files are documentation snapshots;
they neither launch benchmarks nor update the recorded experiment reports.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "metal-method-evidence"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    reports = sorted({p for base in (ROOT / "results", ROOT / "test2026/metal/results")
                      for p in base.rglob("README.md")})
    records = []
    for path in reports:
        raw = path.read_bytes()
        content = raw.decode("utf-8-sig")
        relative = path.relative_to(ROOT).as_posix()
        heading = next((s.lstrip("# ") for s in content.splitlines() if s.startswith("# ")),
                       path.parent.name)
        artifacts = sorted(p.name for p in path.parent.iterdir() if p.is_file())
        verification = []
        for p in sorted(path.parent.glob("*verification*.json")):
            data = p.read_bytes()
            # Keep original verifier bytes as the authority; do not infer success
            # from a file's existence or from its name.
            verification.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(data)))
        records.append(dict(
            experiment_id=path.parent.relative_to(ROOT).as_posix(),
            title=heading, report_path=relative, report_sha256=sha(raw),
            classification="historical_report_requires_scope_and_status_review",
            report_text=content,
            commands_as_documented=re.findall(r"```(?:text|bash|sh|powershell|python)?\s*\n(.*?)```", content, re.S),
            archive_hashes_as_documented=sorted(set(re.findall(r"\b[a-f0-9]{64}\b", content))),
            adjacent_files=artifacts, verification_records=verification,
            evidence_directories=sorted(p.name for p in path.parent.iterdir() if p.is_dir()),
        ))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    document = dict(
        schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=commit,
        scope="All experiment README reports currently present under repository-root results and test2026/metal/results.",
        limitations=["Inventory presence does not establish success or independent replication.",
                     "Reports retain their original historical status; live-state changes belong in the research journal.",
                     "Documented hashes are copied as source material, not newly revalidated archive hashes.",
                     "Command excerpts may contain ROOT/ARCHIVE placeholders; use the archived executed commands for reproduction.",
                     "Unreported experiments and unavailable remote-only artifacts are outside this inventory."],
        report_count=len(records), reports=records,
    )
    (OUTPUT / "index.json").write_bytes((json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    buf = io.StringIO(newline="")
    fields = ["experiment_id", "title", "report_path", "report_sha256", "classification", "verification_paths"]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for r in records:
        writer.writerow({**{k: r[k] for k in fields[:-1]},
                         "verification_paths": ";".join(v["path"] for v in r["verification_records"])})
    (OUTPUT / "index.csv").write_bytes(buf.getvalue().encode("utf-8"))
    lines = ["# Metal experiment evidence index", "",
             f"Generated UTC: {document['generated_at_utc']}", "",
             f"Source commit: `{commit}`. Reports indexed: **{len(records)}**.", "",
             "Each row links to its original report. Read that report's scope, exclusions and status before citing it. "
             "The JSON snapshot retains complete report text, documented command blocks and hashes; the CSV supports analysis. "
             "Neither report presence nor a verification filename certifies a successful experiment.", "",
             "| Experiment | Original report title |", "| --- | --- |"]
    for r in records:
        target = "../../../" + r["report_path"]
        title = r["title"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| [{r['experiment_id']}]({target}) | {title} |")
    (OUTPUT / "INDEX.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(json.dumps(dict(reports=len(records), output=str(OUTPUT), source_commit=commit)))


if __name__ == "__main__":
    main()
