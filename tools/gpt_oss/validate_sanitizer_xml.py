"""Reject every sanitizer record except exact documented initialization statuses.

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
    callers = {('cudaFuncGetAttributes', '209'): ('ncclInitKernelsForDevice',),
               ('cudaGetLastError', '209'): ('ncclInitKernelsForDevice',),
               ('cudaGetLastError', '704'): ('p2pRecvConnect', 'p2pMap', 'p2pMap [clone .isra.7]')}.get((api, code))
    if callers is None:
        return None
    stack = record.find('hostStack')
    if stack is None or stack.findtext('saveLocation') != 'error':
        return None
    frames = stack.findall('frame')
    if len(frames) < 3 or frames[1].findtext('func') != api:
        return None
    if any(Path(frame.findtext('module', '')).name != 'libnccl.so.2' for frame in frames[1:3]):
        return None
    for caller in callers:
        if re.fullmatch(re.escape(caller) + r'(?:\(.*\))?', frames[2].findtext('func', '')):
            return f'{api}/{code}/{caller}'
    return None


def known_initialization(record):
    known = known_nccl_initialization(record)
    if known:
        return known
    # TensorToDevice checks EnablePeerAccess's return before clearing only704.
    # CUDA documents this as an already-established connection, not a failed copy.
    # Do not exclude704 from any other Garnet caller, or any other status here.
    if (record.findtext('kind'), record.findtext('what/api'), record.findtext('what/result')) != ('Api', 'cudaGetLastError', '704'):
        return None
    stack = record.find('hostStack')
    if stack is None or stack.findtext('saveLocation') != 'error':
        return None
    frames = stack.findall('frame')
    if len(frames) < 3 or frames[1].findtext('func') != 'cudaGetLastError':
        return None
    if (Path(frames[1].findtext('module', '')).name != 'libcudart.so.13' or
            Path(frames[2].findtext('module', '')).name != 'libgarnet.so' or
            not re.fullmatch(r'Garnet::GarnetAPI::TensorToDevice(?:\(.*\))?', frames[2].findtext('func', ''))):
        return None
    return 'cudaGetLastError/704/Garnet::GarnetAPI::TensorToDevice'


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
        known = known_initialization(record)
        if known:
            excluded[known] += 1
        else:
            unexpected.append(dict(index=index, kind=record.findtext('kind'),
                api=record.findtext('what/api'), result=record.findtext('what/result'),
                text=record.findtext('what/text')))
    audit = dict(xml=str(source.resolve()), sha256=hashlib.sha256(data).hexdigest(),
        records=len(records), excluded_known_initialization=dict(excluded),
        unexpected=unexpected, passed=not unexpected,
        scope='Only exact NCCL initialization probes and documented704 clearing in Garnet TensorToDevice are excluded; all device-memory and other API reports are fatal. Kernel instrumentation coverage is a separate requirement.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2))
    if unexpected:
        raise RuntimeError(f'{len(unexpected)} unexpected sanitizer records; evidence retained in {output}')
    print(f'Sanitizer gate: zero unexpected records; {sum(excluded.values())} documented initialization API records excluded', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Expected COMPLETE_XML NEW_AUDIT_JSON')
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
