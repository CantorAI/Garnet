"""Exact observed native BF16 control probes; independent mutation hazards."""
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/gpt_oss'))
from validate_sanitizer_xml import known_initialization, known_nccl_graph_initialization

assert len(sys.argv)==2
records=ET.parse(sys.argv[1]).getroot().findall('record')
probes=[r for r in records if r.findtext('what/api')=='cuMemRetainAllocationHandle']
assert len(records)==56 and len(probes)==2 and all(known_initialization(r) for r in records)
assert all(known_nccl_graph_initialization(r)=='cuMemRetainAllocationHandle/1/NCCL-IPC-graph-init/original-bf16-control' for r in probes)
rejected=0
for source in probes:
    def corrupt(path,value):
        global rejected
        candidate=copy.deepcopy(source);candidate.find(path).text=value
        assert known_initialization(candidate) is None,(path,value);rejected+=1
    for path,value in (('kind','InvalidGlobalRead'),('what/api','cuMemMap'),('what/result','2'),
                       ('what/error','CUDA_ERROR_OUT_OF_MEMORY'),('what/message','other'),
                       ('hostStack/saveLocation','launch')):
        corrupt(path,value)
    for index in range(11):
        corrupt(f'hostStack/frame[{index+1}]/module','/tmp/unrelated.so')
        corrupt(f'hostStack/frame[{index+1}]/func','unrelatedCaller')
    for index in (1,2,3):
        corrupt(f'hostStack/frame[{index+1}]/path','other/p2p.cc')
        corrupt(f'hostStack/frame[{index+1}]/line','0')
    corrupt('hostStack/frame[10]/func','Garnet::GptOssTpAllReduce')
    corrupt('hostStack/frame[10]/func','Garnet::GptOssTpAllGather')
    corrupt('hostStack/frame[10]/module','/tmp/libgarnet_gpt_oss.so')
    for index in (9,10):
        candidate=copy.deepcopy(source);candidate.find('hostStack').remove(candidate.findall('hostStack/frame')[index])
        assert known_initialization(candidate) is None;rejected+=1
print(f'All56 actual API records exactly scoped; {rejected} independent hazards remain fatal PASS')
