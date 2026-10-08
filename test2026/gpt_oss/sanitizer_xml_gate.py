"""Check the strict gate against recorded native NCCL XML and injected hazards."""
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/gpt_oss'))
from validate_sanitizer_xml import known_initialization

records=[record for source in sys.argv[1:] for record in ET.parse(source).getroot().findall('record')]
assert records and all(known_initialization(record) for record in records)
cases={known_initialization(record): record for record in records}
hazards = 0
for record in cases.values():
    for path,value in [('kind','Memory'),('what/api','cudaLaunchKernel'),('what/result','719'),
                       ('hostStack/saveLocation','launch'),
                       ('hostStack/frame[2]/func','cudaMemcpyPeer'),
                       ('hostStack/frame[2]/module','/tmp/libunrelated.so'),
                       ('hostStack/frame[3]/func','Garnet::GptOssTpPackBf16'),
                       ('hostStack/frame[3]/module','/tmp/libunrelated.so')]:
        hazard=deepcopy(record)
        field=hazard.find(path)
        assert field is not None
        field.text=value
        assert known_initialization(hazard) is None, f'Unexpected exclusion: {path}={value}'
        hazards += 1
    if 'GptOssTpDirectAcquire' in known_initialization(record):
        for path,value in [('what/error','cudaErrorLaunchFailure'),
                           ('hostStack/frame[4]/func','Garnet::UnrelatedAcquire'),
                           ('hostStack/frame[4]/module','/tmp/libunrelated.so')]:
            hazard=deepcopy(record); hazard.find(path).text=value
            assert known_initialization(hazard) is None, (path,value)
            hazards += 1
print(f'All {len(records)} recorded initialization APIs classified; {hazards} memory/API/code/caller/module hazards rejected')
