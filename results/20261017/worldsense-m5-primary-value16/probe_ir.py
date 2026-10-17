"""CPU compiler diagnostic, not GPU execution or inference performance."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Exact helper fragment in dynamic standalone Metalprobe, CLI defaults withO3. Runtime selectedprofile uses nilMTLCompileOptions; this does not claim whole-runtime-kernel or GPUmachinecode identity, registerusage, timing or causal speed.',commands=[],versions={},variants={})
for label,cmd in [('compiler',['xcrun','-sdk','macosx','metal','--version']),('path',['xcrun','--find','metal']),('sdk',['xcrun','-sdk','macosx','--show-sdk-version'])]:
    r=subprocess.run(cmd,capture_output=True,text=True);report['versions'][label]=dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
for mode,helper in [('off','legacy-helper.txt'),('on','candidate-helper.txt')]:
    source=w/('probe-'+mode+'.metal');output=w/('probe-'+mode+'.ll')
    source.write_text('#include <metal_stdlib>\nusing namespace metal;\n'+(w/helper).read_text()+'''
kernel void value16_probe(device const uint* input [[buffer(0)]],device ushort4* output [[buffer(1)]],uint id [[thread_position_in_grid]]) {
 output[id]=primary_quad_kword(input[id*3],ushort(input[id*3+1]),input[id*3+2]);
}
''')
    cmd=['xcrun','-sdk','macosx','metal','-O3','-S','-emit-llvm',str(source),'-o',str(output)]
    r=subprocess.run(cmd,capture_output=True,text=True);report['commands'].append(cmd)
    report['variants'][mode]=dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256=sha(source))
    if not r.returncode:report['variants'][mode]['IR_sha256']=sha(output)
def normalize(raw):
    definitions=re.findall(r'^define\b.*?^}',raw,re.M|re.S)
    assert definitions,'No LLVMfunction definitions; cannot call modules identical'
    text='\n'.join(definitions)
    text=re.sub(r';[^\n]*','',text)
    text=re.sub(r'!\w+\s+!\d+','',text)
    names={}
    def number(match):
        token=match.group(0)
        if token not in names:names[token]='%v'+str(len(names))
        return names[token]
    text=re.sub(r'%[A-Za-z0-9_.]+',number,text)
    return '\n'.join(' '.join(line.split())for line in text.splitlines()if line.strip())
if all(v['exit']==0 for v in report['variants'].values()):
    normalized={mode:normalize((w/('probe-'+mode+'.ll')).read_text())for mode in ['off','on']}
    for mode,text in normalized.items():
        (w/('probe-'+mode+'-normalized.txt')).write_text(text)
        report['variants'][mode]['normalized_function_sha256']=hashlib.sha256(text.encode()).hexdigest()
        report['variants'][mode]['integer_opcode_counts']={op:len(re.findall(r'\b'+op+r'\b',text))for op in ['shl','lshr','or','and','mul','trunc','zext']}
    identical=normalized['off']==normalized['on']
    report.update(normalized_helper_probe_IR_identical=identical,performance_eligible=not identical,decision='exclude_before_performance_standalone_compiled_noop'if identical else'distinct_standalone_integer_IR_requires_full_native_GPU_correctness_and_fresh_gates')
else:
    report.update(normalized_helper_probe_IR_identical=None,performance_eligible=None,decision='Compilerdiagnosticfailed; retainerror and generated-code uncertainty; no claim of mechanism or performance')
(w/'compiler-IR-diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items()if k not in ['commands','versions']},indent=2))
