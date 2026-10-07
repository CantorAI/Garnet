"""Check the strict gate against recorded native NCCL XML and injected hazards."""
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/gpt_oss'))
from validate_sanitizer_xml import known_nccl_initialization

records=ET.parse(sys.argv[1]).getroot().findall('record')
assert records and all(known_nccl_initialization(record) for record in records)
for path,value in [('kind','Memory'),('what/api','cudaLaunchKernel'),('what/result','719'),
                   ('hostStack/frame[3]/func','Garnet::GptOssTpPackBf16'),
                   ('hostStack/frame[3]/module','/tmp/libgarnet_gpt_oss.so')]:
    hazard=deepcopy(records[0])
    field=hazard.find(path)
    assert field is not None
    field.text=value
    assert known_nccl_initialization(hazard) is None, f'Unexpected exclusion: {path}={value}'
print(f'All {len(records)} recorded NCCL initialization APIs classified; five memory/API/code/caller hazards rejected')
