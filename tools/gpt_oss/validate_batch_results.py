"""Python: validate_batch_results.py MODEL RESULT_JSON EXPECTED_JSON OUTPUT_JSON.

Check the first final-channel JSON from every fixed-output batch slot. Tokens
generated after the answer are retained in the throughput result but ignored
for the task-answer check.
"""
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer


assert len(sys.argv) == 5, 'expected model, result, expected, output'
model, result_path, expected_path, output_path = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text())
expected = json.loads(expected_path.read_text())
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
outcomes = []
for slot, ids in enumerate(result['token_ids_by_request']):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    matches = re.findall(r'<\|channel\|>final<\|message\|>\s*(\{.*?\})',
                         text, re.DOTALL)
    actual = None
    if matches:
        try:
            actual = json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    outcomes.append({'slot': slot, 'pass': actual == expected,
                     'first_final_json': actual})

output_path.write_text(json.dumps({'result': str(result_path),
                                   'all_pass': all(row['pass'] for row in outcomes),
                                   'slots': outcomes}, indent=2))
print(output_path, 'PASS' if all(row['pass'] for row in outcomes) else 'FAIL',
      flush=True)
if not all(row['pass'] for row in outcomes):
    raise SystemExit(1)
