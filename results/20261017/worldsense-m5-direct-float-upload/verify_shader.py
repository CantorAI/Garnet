from pathlib import Path
import json,subprocess,hashlib
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-direct-float-upload/source');repo=Path('/Users/shawn/CantorAI/Garnet');base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
old=subprocess.check_output(['git','-C',str(repo),'show',base+':src/backends/metal/primary_block_shader.h']).decode()
assert (s/'src/backends/metal/primary_block_shader.h').read_text()==old
raw=old.split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
new=(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
preamble=(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PMIN(',1)[1].split(')PMIN"',1)[0]
helper_start=old.index('inline ushort4 primary_quad_kword(');helper_end=old.index('inline PrimaryQuad primary_quad(',helper_start)
abi='struct BF16Params {ulong rows,inner,columns,aStride,oStride;};'
assert preamble=='#include <metal_stdlib>\nusing namespace metal;\n'+abi+'\n'+old[helper_start:helper_end]
assert abi in (s/'src/backends/metal/metal_context.mm').read_text()
(w/'minimal-preamble.metal').write_text(preamble)
specs=[('primary_bf16_gemm','primary_vector_bf16_gemm','32*32',32,32,128),('primary_gemm_16_64_64','primary_vector_gemm_16_64_64','4096',64,64,256),('primary_gemm_32_64_32','primary_vector_gemm_32_64_32','2048',64,32,256)]
checks=[]
for name,newName,total,c,k,threads in specs:
 start=raw.index('kernel void '+name+'(');end=raw.find('kernel void ',start+1);before=raw[start:end if end>=0 else len(raw)]
 start=new.index('kernel void '+newName+'(');end=new.find('kernel void ',start+1);after=new[start:end]
 if newName==specs[-1][1]:after=after.split('struct PrimaryPrefillTileProof',1)[0]
 # The appended fragment separator contributes only blank trailing whitespace.
 before=before.rstrip();after=after.rstrip()
 previous=f'for(uint i=tid;i<{total};i+={threads}){{ulong k=base+i/{c},column=colBase+i%{c};right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16):0.0f;}}'
 current=f'primary_prefill_tile_fill<{c},{k},{threads}>(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),tid);'
 declarations={
 'primary_bf16_gemm':('threadgroup float left[16*32],right[32*32],result[16*32];','threadgroup float left[16*32],result[16*32];\n    threadgroup float4 rightAligned[256];threadgroup float* right=(threadgroup float*)rightAligned;'),
 'primary_gemm_16_64_64':('threadgroup float left[1024],right[4096],result[1024];','threadgroup float left[1024],result[1024];\nthreadgroup float4 rightAligned[1024];threadgroup float* right=(threadgroup float*)rightAligned;'),
 'primary_gemm_32_64_32':('threadgroup float left[1024],right[2048],result[2048];','threadgroup float left[1024],result[2048];\nthreadgroup float4 rightAligned[512];threadgroup float* right=(threadgroup float*)rightAligned;')}
 oldDeclaration,newDeclaration=declarations[name];assert after.count(newDeclaration)==1
 after=after.replace(newDeclaration,oldDeclaration,1)
 assert after.count(current)==1
 assert after.replace('kernel void '+newName+'(','kernel void '+name+'(',1).replace(current,previous,1)==before
 checks.append(dict(kernel=newName,entire_body_inverse_recovered_except_fragment_trailing_blanks=True))
cpp='''#include "primary_block_shader.h"
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
#include "primary_vector_prefill_shader.h"
#endif
#include <fstream>
int main(int argc,char**argv){std::ofstream out(argv[1],std::ios::binary);out<<Garnet::kPrimaryBlockShader;
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
out<<Garnet::kPrimaryVectorPrefillShader;
#endif
return !out;}
'''
(w/'materialize_shader.cpp').write_text(cpp);commands=[]
for mode in ['off','on']:
 exe=w/('materialize-'+mode);cmd=['/usr/bin/c++','-std=c++17','-O0','-I',str(s/'src/backends/metal'),str(w/'materialize_shader.cpp'),'-o',str(exe)]
 if mode=='on':cmd+=['-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=1']
 subprocess.run(cmd,check=True);out=w/('materialized-primary-'+mode+'.metal');subprocess.run([str(exe),str(out)],check=True);commands.append(cmd)
 assert out.read_text()==raw+(new if mode=='on'else'')
report={'immutable_base':base,'original_primary_header_byte_exact':True,'minimal_preamble_ABI_and_helper_exact':True,'CXX_materialized_OFF_primary_fragment_exact':True,'CXX_materialized_ON_equals_original_plus_new_fragment':True,'kernel_inverse_checks':checks,'commands':commands,'scope':'Actual C++ header fragment/name/body proof, fragment/alignment-declaration substitution only; actual runtime main-program proof requires subsequent native constructor dumps. Not GPU machine code or performance.'}
(w/'independent-shader-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
