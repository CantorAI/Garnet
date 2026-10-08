"""Independent raw native XQA coverage/status/identity audit, not serving proof.

No adapter/page-contract/source-generator/sanitizer implementation imports.
The FP64 values are sampled and checked internally by the captured fixture;
this auditor verifies independent coverage counters and recorded status, not
unarchived full numerical matrices. DIRECTORY NEW_REPORT.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

PINNED_SUMS='e400f9df3446f52f3cb70f3fd0af812fa322bd25edf5c0c27e9070b23991bdb1'
VENDOR_METADATA={'.gitattributes':'609e4a49dac664d8f6b3fddd805b2dbcb59ab6d2bf8ca17ff701ffac6fee7014',
 'PROVENANCE.txt':'d06b8134138e33a2e4e0fdfed8c00d698961505ae059e1e3dba5f58485c3f6db'}
# B, logical pages, window, KV heads, layer, page pattern, unique pages, fixed length.
SPECS=[(1,16,0,1,0,1,False,1),(1,16,23,4,1,2,False,17),
 (1,160,0,4,0,1,False,2005),(1,160,128,4,1,3,False,2005),
 (1,256,0,16,1,4,False,4096),(1,16,0,1,0,1,False,-1),
 (16,16,0,4,0,0,False,0),(16,160,23,1,1,0,False,0),(16,256,128,16,0,0,False,0),
 (128,160,0,4,0,0,False,0),(128,160,128,4,1,0,False,0),
 (144,160,0,4,1,0,False,0),(144,256,23,4,0,0,False,0),
 (448,16,0,4,0,1,True,0),(448,160,128,4,1,0,False,0),
 (512,160,0,4,0,0,False,0),(512,256,23,4,1,0,False,0),(512,256,128,16,1,0,False,0)]

def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def coverage():
    totals=dict(cases=36,invocations=144,rows=0,table_entries=0,sampled_fp64_values=0,zero_values=0,guards=36*31)
    for b,pages,window,heads,layer,pattern,unique,fixed in SPECS:
        samples={0,b//2,b-1}|set(range(min(b,14)))
        totals['rows']+=b*4*2;totals['table_entries']+=b*pages*4*2
        totals['sampled_fp64_values']+=len(samples)*heads*8*64*4*2
        lengths=[1,15,16,17,127,128,129,min(2005,pages*16),pages*16-1,pages*16,0,-1,pages*16+1,2**31-1]
        for revision in (0,1):
            for row in range(b):
                end=fixed if fixed else lengths[(row+revision*3)%14]
                if not fixed and row%29==28:end=-(2**31)
                if row%19==18 or not 0<end<=pages*16:
                    totals['zero_values']+=heads*8*64*2*2
    return totals

def audit_log(path):
    lines=Path(path).read_text().splitlines();device=None;cases=[];summary=[]
    for line in lines:
        if line.startswith('DEVICE '):
            match=re.fullmatch(r'DEVICE ([01]) sequential independent native tests',line)
            require(match is not None,'device line');device=int(match[1])
        elif line.startswith('CASE '):
            match=re.fullmatch(r'CASE (\d+) B=(\d+) KVH=(\d+) pages=(\d+) physical=(\d+) window=(\d+) layer=(\d+) pattern=(\d+) ordinary/captured PASS',line)
            require(match is not None and device is not None,'case line');index=int(match[1])
            require(0<=index<18,'case index');b,p,w,k,l,pattern,unique,fixed=SPECS[index]
            require(list(map(int,match.groups()[1:]))==[b,k,p,b*p if unique else 9,w,l,pattern],'case dimensions')
            cases.append((device,index))
        elif line.startswith('XQA_NATIVE_PARITY_ACTUAL_PASS '):summary.append(line)
    require(cases==[(d,i) for d in (0,1) for i in range(18)],'all36 ordered cases on both GPUs')
    require(len(summary)==1,'exactly one complete summary')
    match=re.fullmatch(r'XQA_NATIVE_PARITY_ACTUAL_PASS devices=2 cases=(\d+) invocations=(\d+) rows=(\d+) table_entries=(\d+) sampled_fp64_values=(\d+) zero_values=(\d+) guards=(\d+) max_abs=([^ ]+) bound=\.006\*\(1\+abs\)',summary[0])
    require(match is not None,'unweakened summary schema')
    expected=coverage();require(dict(zip(expected,map(int,match.groups()[:7])))==expected,'independent coverage counters')
    maximum=float(match[8]);require(math.isfinite(maximum) and 0<=maximum<=.006*(1+.1875),'reported finite sampled error upper bound')
    require(any(line.startswith('SCOPE native sampled FP64/full safety metadata/read-only KV/ordinary+changed graph; NOT compiled/plugin/pretrained/performance') for line in lines),'declared sampled native-only scope')
    require(not any('XQA_NATIVE_PARITY_FAILED' in line or 'XQA cleanup failed:' in line for line in lines),'application/cleanup failure')
    return dict(coverage=expected,reported_max_abs=maximum,sha256=sha(path),
        limits='Internal sampled FP64 checks; no archived full output/reference matrices or serving claim')

def file_set(folder):
    paths=list(Path(folder).iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in paths),'source closure must be regular files')
    return {p.name:sha(p) for p in paths}

def audit(root,output):
    root,output=Path(root),Path(output);require(not output.exists(),'refuse audit overwrite')
    terminal=read(root/'terminal.json')
    require(terminal['terminal'] is True and terminal['actual_exit']==0 and terminal['phase']=='complete-native-only','actual complete wrapper/application status')
    metadata=read(root/'gate-metadata.json')
    require(re.fullmatch('[0-9a-f]{40}',metadata['source_commit']) and (root/'source.txt').read_text().strip()==metadata['source_commit'],'source commit')
    require(sha(root/'native-payload/xqa-parity')==metadata['private_binary_sha256'],'private binary')
    require(set(metadata['inference_binaries'])=={'libgarnet.so','libgarnet_gpt_oss.so','garnet_gpt_oss_kernel_parity'},'inference identity keys')
    for name,digest in metadata['inference_binaries'].items():require(sha(root/'native-payload'/name)==digest,'inference payload changed')
    for name,digest in metadata['source_sha256'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'source path')
        require(sha(root/'source'/relative)==digest,'fixture/adapter/generator source changed')
    required={'test2026/gpt_oss/xqa_attention_parity.cu','plugins/gpt_oss/cuda/gpt_oss_xqa_attention.cu',
        'plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.cu','plugins/gpt_oss/include/gpt_oss_xqa_attention.h',
        'plugins/gpt_oss/include/gpt_oss_xqa_layout.h','plugins/gpt_oss/cuda/gpt_oss_xqa_bridge.h',
        'plugins/gpt_oss/include/gpt_oss_kernels.h','plugins/gpt_oss/CMakeLists.txt','tools/gpt_oss/prepare_xqa_sources.py'}
    require(set(metadata['source_sha256'])==required,'complete own-source closure')
    vendor=file_set(root/'vendor');require(vendor['SHA256SUMS']==PINNED_SUMS,'pinned upstream manifest')
    records={name:digest for digest,name in (line.split('  ',1) for line in (root/'vendor/SHA256SUMS').read_text().splitlines())}
    require(len(records)==17 and set(vendor)==set(records)|{'SHA256SUMS'}|set(VENDOR_METADATA) and
        all(vendor[n]==h for n,h in {**records,**VENDOR_METADATA}.items()),'immutable17 originals/license/notice and recorded provenance')
    generated=file_set(root/'generated');adaptation=read(root/'generated/garnet-adaptation.json')
    require(generated==metadata['generated_sha256'] and adaptation['upstream_manifest_sha256']==PINNED_SUMS and
        adaptation['upstream_commit']=='946200de1ae94fc93fdd0926f0a13afd1fa7f0f1','generated closure identity')
    require(set(generated)==set(records)|{'garnet-adaptation.json'} and
        adaptation['generated_sha256']=={n:h for n,h in generated.items() if n!='garnet-adaptation.json'},'complete generated18 byte manifest')
    for name in ('LICENSE','NOTICE'):require(generated[name]==records[name],'upstream license/notice preserved')
    hardware=list(csv.DictReader((root/'hardware.csv').read_text().splitlines(),skipinitialspace=True))
    require(len(hardware)==2 and [r['index'] for r in hardware]==['0','1'] and len({r['uuid'] for r in hardware})==2,'both distinct recorded GPUs')
    logs={n:audit_log(root/(n+'.log')) for n in ('native','native-mem','native-trace')}
    xml=root/'native-mem.xml';tree=ET.fromstring(xml.read_bytes())
    require(tree.tag=='ComputeSanitizerOutput' and not tree.findall('record'),'standalone native requires zero records, no exclusions')
    mem=(root/'native-mem.log').read_text()
    require(re.search(r'ERROR SUMMARY:\s*0 errors',mem),'complete zero-error tool summary')
    require(not any(x in mem for x in ('No attachable process found','compute-sanitizer timed-out','========= Error:','Target application returned an error')),'tool/application coverage failure')
    counts=Counter();header=None
    for row in csv.reader((root/'native-kernels.csv').read_text().splitlines()):
        if 'Name' in row and 'Instances' in row:header=row;continue
        if header and len(row)==len(header):
            for kernel in ('GarnetGptOssXqaKernel','gptOssXqaPrepare','gptOssXqaFinalize'):
                if kernel in row[header.index('Name')]:counts[kernel]+=int(row[header.index('Instances')])
    require(all(counts[n]>=144 for n in ('GarnetGptOssXqaKernel','gptOssXqaPrepare','gptOssXqaFinalize')),'all ordinary/changed replay native dispatch')
    report=dict(scope='Private native coverage/identity/unrestricted memory/dispatch; NOT compiled/pretrained/performance/full-goal proof',
        source_commit=metadata['source_commit'],logs=logs,inference_binaries=metadata['inference_binaries'],
        private_binary_sha256=metadata['private_binary_sha256'],generated_sha256=generated,
        sanitizer=dict(records=0,exclusions=0,unexpected=0,sha256=sha(xml)),dispatch=dict(counts))
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print('INDEPENDENT_XQA_NATIVE_RAW_AUDIT_PASS',coverage(),flush=True)
    return report

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Expected COMPLETE_DIRECTORY NEW_REPORT')
    audit(sys.argv[1],sys.argv[2])
