"""Re-audit a complete saved XML and inject independent graph-probe hazards."""
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/gpt_oss'))
from validate_sanitizer_xml import known_initialization, known_nccl_graph_initialization

assert 1<=len(sys.argv[1:])<=3
sources=[]; total=0
for path in sys.argv[1:3]:
    records=ET.parse(path).getroot().findall('record')
    probes=[r for r in records if r.findtext('what/api')=='cuMemRetainAllocationHandle']
    assert len(probes)==576 and all(known_nccl_graph_initialization(r) for r in probes)
    assert len(records)==686 and all(known_initialization(r) for r in records)
    sources.append(probes[0]);total+=len(records)
if len(sources)==2:
    assert {known_nccl_graph_initialization(r).rsplit('-',1)[1] for r in sources}=={'fp32','bf16'}
rejected=0
def corrupt(path,value):
    global rejected
    candidate=copy.deepcopy(source)
    candidate.find(path).text=value
    assert known_initialization(candidate) is None,(path,value)
    rejected+=1
for source in sources:
    for path,value in (('kind','InvalidGlobalRead'),('what/api','cuMemMap'),('what/result','2'),
                       ('what/error','CUDA_ERROR_OUT_OF_MEMORY'),('what/message','other'),
                       ('hostStack/saveLocation','launch')):
        corrupt(path,value)
    for index in range(10):
        corrupt(f'hostStack/frame[{index+1}]/module','/tmp/unrelated.so')
        corrupt(f'hostStack/frame[{index+1}]/func','unrelatedCaller')
    for index in (1,2,3):
        corrupt(f'hostStack/frame[{index+1}]/path','other/p2p.cc')
        corrupt(f'hostStack/frame[{index+1}]/line','0')
    corrupt('hostStack/frame[10]/func','Garnet::GptOssTpAllGather')
    corrupt('hostStack/frame[10]/module','/tmp/libgarnet_gpt_oss.so')
    corrupt('hostStack/frame[10]/module','/tmp/garnet_gpt_oss_tp_direct_benchmark')
    short=copy.deepcopy(source)
    short.find('hostStack').remove(short.findall('hostStack/frame')[9])
    assert known_initialization(short) is None
    rejected+=1
if len(sys.argv)==4:
    malformed=ET.parse(sys.argv[3]).getroot().findall('record')
    probes=[r for r in malformed if r.findtext('what/api')=='cuMemRetainAllocationHandle']
    assert len(malformed)==686 and len(probes)==576
    assert all(known_initialization(r) is None for r in probes)
    assert sum(known_initialization(r) is None for r in malformed)==576
print(f'All{total} saved well-formed API records scoped; {rejected} independent hazards rejected; malformed records stay fatal PASS')
