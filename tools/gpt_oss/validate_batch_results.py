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

if len(sys.argv)==3 and sys.argv[1]=='--preflight':
    AutoTokenizer.from_pretrained(Path(sys.argv[2]), local_files_only=True)
    print('Batch validator dependency/local-tokenizer preflight PASS',flush=True)
    raise SystemExit(0)

assert len(sys.argv) == 5, 'expected model, result, expected, output'
model, result_path, expected_path, output_path = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text())
expected = json.loads(expected_path.read_text())
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
trials = result.get('decode_trials', [result])
assert trials, 'no decode trials to validate'
outcomes = []
for trial, data in enumerate(trials):
    assert len(data['token_ids_by_request']) == result['batch'], 'missing batch slots'
    for slot, ids in enumerate(data['token_ids_by_request']):
        assert len(ids) == result['output_tokens_per_request'], 'wrong fixed output length'
        text = tokenizer.decode(ids, skip_special_tokens=False)
        suffix = f'.trial{trial}' if len(trials) > 1 else ''
        decoded_path = output_path.with_name(output_path.stem + suffix + f'.slot{slot}.txt')
        decoded_path.write_text(text)
        matches = re.findall(r'<\|channel\|>final<\|message\|>\s*(\{.*?\})',
                             text, re.DOTALL)
        actual = None
        if matches:
            try:
                actual = json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        outcomes.append({'trial': trial, 'slot': slot, 'pass': actual == expected,
                         'decoded_path': str(decoded_path),
                         'first_final_json': actual})

output_path.write_text(json.dumps({'result': str(result_path),
                                   'all_pass': all(row['pass'] for row in outcomes),
                                   'slots': outcomes}, indent=2))
print(output_path, 'PASS' if all(row['pass'] for row in outcomes) else 'FAIL',
      flush=True)
if not all(row['pass'] for row in outcomes):
    raise SystemExit(1)
