"""CPU preflight and evidence paths for an explicit same-shape resident session."""
import hashlib
import json
from pathlib import Path


def case_artifacts(result):
    result = Path(result)
    return [result, result.with_suffix('.trials.partial.json'),
            result.with_suffix('.memory-failure.json')]


def session_partial(result):
    return Path(result).with_suffix('.session.partial.json')


def read_session(manifest_path, summary_path, batch, output, capacity, chunk,
                 protected_paths=()):
    manifest_path, summary_path = Path(manifest_path), Path(summary_path)
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if (manifest.get('resident_session_schema') != 1 or
            any(type(manifest.get(key)) is not int or manifest[key] != value
                for key, value in [('batch', batch), ('output', output),
                                   ('context', capacity), ('prefill_chunk', chunk)]) or
            not isinstance(manifest.get('cases'), list) or
            not 1 <= len(manifest['cases']) <= 32):
        raise ValueError('Invalid resident session schema or shared workload')
    validator_python, tokenizer = Path(manifest['validation_python']), Path(manifest['tokenizer'])
    if not validator_python.is_absolute() or not validator_python.is_file() or not tokenizer.is_absolute():
        raise ValueError('Session requires an explicit host Python and tokenizer path')
    tokenizer_path = tokenizer / 'tokenizer.json'
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    cases, names = [], set()
    for item in manifest['cases']:
        name = item.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Session case names must be nonempty and unique')
        names.add(name)
        request_path, result_path = Path(item['request']), Path(item['result'])
        expected_path, validation_path = Path(item['expected']), Path(item['validation'])
        if any(not path.is_absolute() for path in (request_path, result_path, expected_path, validation_path)):
            raise ValueError('Session request/result paths must be absolute')
        expected_sha256 = hashlib.sha256(expected_path.read_bytes()).hexdigest()
        request_bytes = request_path.read_bytes()
        request = json.loads(request_bytes)
        ids = request.get('input_ids')
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError('Session request requires positive-length integer token IDs')
        cases.append(dict(name=name, request_path=request_path, result_path=result_path,
            expected_path=expected_path, expected_sha256=expected_sha256,
            validation_path=validation_path,
            request=request, input_ids=ids,
            request_sha256=hashlib.sha256(request_bytes).hexdigest()))
    def identity(case):
        request = case['request']
        return (len(case['input_ids']), request.get('device_ids'),
                request.get('reserve_mb', 1024), request.get('memory_fraction', .9))
    if any(identity(case) != identity(cases[0]) for case in cases[1:]):
        raise ValueError('Resident session input length/device/reserve/memory profile must match')
    protected = {Path(path).resolve() for path in protected_paths}
    protected.add(manifest_path.resolve())
    protected.update(case['request_path'].resolve() for case in cases)
    protected.update(case['expected_path'].resolve() for case in cases)
    protected.update([validator_python.resolve(), tokenizer_path.resolve()])
    outputs = [summary_path, session_partial(summary_path)]
    for case in cases:
        outputs.extend(case_artifacts(case['result_path']))
        outputs.append(case['validation_path'])
        if list(case['validation_path'].parent.glob(case['validation_path'].stem + '*.slot*.txt')):
            raise FileExistsError('Refusing to overwrite previous decoded answer evidence')
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved) or protected.intersection(resolved):
        raise ValueError('Session evidence paths alias an input or another output')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite session evidence: {path}')
    return cases, dict(resident_session_schema=1,
        session_manifest=str(manifest_path.resolve()),
        session_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        session_result=str(summary_path.resolve()), case_order=[case['name'] for case in cases],
        validation_python=str(validator_python), tokenizer=str(tokenizer),
        tokenizer_sha256=tokenizer_sha256)
