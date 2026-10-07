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
for record in cases.values():
    for path,value in [('kind','Memory'),('what/api','cudaLaunchKernel'),('what/result','719'),
                       ('hostStack/saveLocation','launch'),
                       ('hostStack/frame[2]/func','cudaMemcpyPeer'),
                       ('hostStack/frame[2]/module','/tmp/libunrelated.so'),
                       ('hostStack/frame[3]/func','Garnet::GptOssTpPackBf16'),
                       ('hostStack/frame[3]/module','/tmp/libgarnet_gpt_oss.so')]:
        hazard=deepcopy(record)
        field=hazard.find(path)
        assert field is not None
        field.text=value
        assert known_initialization(hazard) is None, f'Unexpected exclusion: {path}={value}'
print(f'All {len(records)} recorded initialization APIs classified; {len(cases)*8} memory/API/code/caller/module hazards rejected')
