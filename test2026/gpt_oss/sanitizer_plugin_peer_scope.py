"""Actual recorded compiled peer initialization and fail-closed mutations."""
from copy import deepcopy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/gpt_oss'))
from validate_sanitizer_xml import known_initialization, known_plugin_peer_initialization

records = ET.parse(sys.argv[1]).getroot().findall('record')
peers = [r for r in records if r.findtext('what/result') == '704']
assert len(records) == 166 and len(peers) == 6
assert all(known_initialization(r) for r in records)
assert {known_plugin_peer_initialization(r) for r in peers} == {
    'cudaGetLastError/704/Garnet::GptOssTpDirectAcquire/compiled-plugin-init',
    'cudaGetLastError/704/Garnet::GptOssTpDirectAcquire/script-group-init'}
hazards = 0
for record in peers:
    fields = [('kind', 'Memory'), ('what/api', 'cudaMemcpyPeer'), ('what/result', '719'),
        ('what/error', 'cudaErrorLaunchFailure'), ('what/message', 'unrelated error'),
        ('hostStack/saveLocation', 'launch')]
    fields += [(f'hostStack/frame[{i}]/module', '/tmp/libunrelated.so') for i in range(1, 7)]
    fields += [(f'hostStack/frame[{i}]/func', 'Garnet::UnrelatedAcquire') for i in range(2, 6)]
    fields += [(f'hostStack/frame[{i}]/pc', '') for i in range(1, len(record.findall('hostStack/frame')) + 1)]
    for path, value in fields:
        candidate = deepcopy(record); field = candidate.find(path); assert field is not None
        field.text = value
        assert known_initialization(candidate) is None, (path, value)
        hazards += 1
    for remaining in range(6):
        candidate = deepcopy(record); stack = candidate.find('hostStack')
        for frame in stack.findall('frame')[remaining:]: stack.remove(frame)
        assert known_initialization(candidate) is None
        hazards += 1
    for path in ('kind', 'what', 'hostStack'):
        candidate = deepcopy(record); candidate.remove(candidate.find(path))
        assert known_initialization(candidate) is None
        hazards += 1
print(f'Actual six initialization records scoped; {hazards} foreign/memory/malformed hazards rejected PASS')
