"""Reject every sanitizer record except three documented NCCL initialization APIs.

Python: COMPLETE_XML NEW_AUDIT_JSON. Use with explicit API reporting, unlimited
records and successful application exit. This does not establish kernel coverage.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def known_nccl_initialization(record):
    if record.findtext('kind') != 'Api':
        return None
    api, code = record.findtext('what/api'), record.findtext('what/result')
    caller = {('cudaFuncGetAttributes', '209'): 'ncclInitKernelsForDevice',
              ('cudaGetLastError', '209'): 'ncclInitKernelsForDevice',
              ('cudaGetLastError', '704'): 'p2pRecvConnect'}.get((api, code))
    if caller is None:
        return None
    stack = record.find('hostStack')
    if stack is None or stack.findtext('saveLocation') != 'error':
        return None
    frames = stack.findall('frame')
    if len(frames) < 3 or frames[1].findtext('func') != api:
        return None
    if any(Path(frame.findtext('module', '')).name != 'libnccl.so.2' for frame in frames[1:3]):
        return None
    if not re.fullmatch(re.escape(caller) + r'(?:\(.*\))?', frames[2].findtext('func', '')):
        return None
    return f'{api}/{code}/{caller}'


def validate(source, output):
    if output.exists():
        raise FileExistsError(output)
    data = source.read_bytes()
    root = ET.fromstring(data)
    if root.tag != 'ComputeSanitizerOutput':
        raise ValueError('Unexpected sanitizer XML schema')
    excluded, unexpected = Counter(), []
    records = root.findall('record')
    for index, record in enumerate(records):
        known = known_nccl_initialization(record)
        if known:
            excluded[known] += 1
        else:
            unexpected.append(dict(index=index, kind=record.findtext('kind'),
                api=record.findtext('what/api'), result=record.findtext('what/result'),
                text=record.findtext('what/text')))
    audit = dict(xml=str(source.resolve()), sha256=hashlib.sha256(data).hexdigest(),
        records=len(records), excluded_known_nccl_initialization=dict(excluded),
        unexpected=unexpected, passed=not unexpected,
        scope='Only exact NCCL initialization API/code/caller frames are excluded; all device-memory and other API reports are fatal. Kernel instrumentation coverage is a separate requirement.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2))
    if unexpected:
        raise RuntimeError(f'{len(unexpected)} unexpected sanitizer records; evidence retained in {output}')
    print(f'Sanitizer gate: zero unexpected records; {sum(excluded.values())} documented NCCL initialization API records excluded', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Expected COMPLETE_XML NEW_AUDIT_JSON')
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
