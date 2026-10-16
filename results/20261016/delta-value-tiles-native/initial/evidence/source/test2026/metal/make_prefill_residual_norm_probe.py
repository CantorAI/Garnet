"""Generate original Add/Layout/RMS control and a three-output packed fusion."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text()
def function(name):
 start=native.index('kernel void '+name+'(');brace=native.index('{',start);depth=1;end=brace+1
 while depth:depth+=(native[end]=='{')-(native[end]=='}');end+=1
 return native[start:end]+'\n'
header='''#include <metal_stdlib>
using namespace metal;
struct Params{ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};
struct LayoutParams{ulong aRows,aColumns,aStride,bStride,oRows,oColumns,oStride,tokenWidth;};
struct PackedParams{ulong rows,columns,packedStride,normStride,chunk;float epsilon,offset;};
'''
fused='''kernel void packed_residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device const float* weight [[buffer(2)]],device float* packed [[buffer(3)]],device float* layout [[buffer(4)]],
 device float* out [[buffer(5)]],constant PackedParams& p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 if(row>=p.rows)return;
 ulong base=(row/p.chunk)*p.packedStride+(row%p.chunk)*p.columns;
 threadgroup float partial[4];float sum=0;
 for(ulong j=tid;j<p.columns;j+=128){float x=a[base+j]+b[base+j];packed[base+j]=x;layout[row*p.normStride+j]=x;sum+=x*x;}
 sum=simd_sum(sum);if(lane==0)partial[simd]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 sum=simd_sum(lane<4?partial[lane]:0.0f);
 float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
 for(ulong j=tid;j<p.columns;j+=128)out[row*p.normStride+j]=layout[row*p.normStride+j]*inverse*(weight[j]+p.offset);
}
'''
Path(sys.argv[2]).write_bytes((header+function('add_matrix')+function('chunk_layout')+function('rms_norm')+fused).encode())
text=Path(__file__).with_name('residual_norm_gpu.mm').read_text()
def replace(old,new,count=1):
 global text
 assert text.count(old)==count,(old,text.count(old));text=text.replace(old,new)
replace('struct Params {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};','''struct Params {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
struct LayoutParams {uint64_t aRows,aColumns,aStride,bStride,oRows,oColumns,oStride,tokenWidth;};
struct PackedParams {uint64_t rows,columns,packedStride,normStride,chunk;float epsilon,offset;};''')
replace('pipe[3];NSString* names[3]={@"add_matrix",@"rms_norm",@"residual_norm"};','pipe[4];NSString* names[4]={@"add_matrix",@"rms_norm",@"packed_residual_norm",@"chunk_layout"};')
replace('i<3;++i){pipe[i]','i<4;++i){pipe[i]')
replace('for(uint64_t rows:{1,2,16,32,64})','for(uint64_t batch:{1,2})for(uint64_t chunk:{8,16,22,26,32,64,80})')
replace('uint64_t stride=((columns+15)/16)*16+16;Params add{rows,columns,stride,stride,stride,0,0},norm{rows,columns,stride,columns,stride,eps,offset};','''uint64_t rows=batch*chunk,stride=((columns+15)/16)*16+16,packedStride=chunk*columns+16;
   Params add{batch,chunk*columns,packedStride,packedStride,packedStride,0,0},norm{rows,columns,stride,columns,stride,eps,offset};
   LayoutParams lp{batch,chunk*columns,packedStride,packedStride,rows,columns,stride,0};
   PackedParams fp{rows,columns,packedStride,stride,chunk,eps,offset};''')
replace('std::vector<float> a(rows*stride),b(rows*stride);for(uint64_t r=0;r<rows;++r)for(uint64_t k=0;k<columns;++k){a[r*stride+k]=std::sin(float(r*columns+k)*.013f)*.3f;b[r*stride+k]=std::cos(float(r*columns+k)*.027f)*.2f;}','''std::vector<float> a(batch*packedStride),b(batch*packedStride);for(uint64_t r=0;r<rows;++r)for(uint64_t k=0;k<columns;++k){uint64_t index=(r/chunk)*packedStride+(r%chunk)*columns+k;a[index]=std::sin(float(r*columns+k)*.013f)*.3f;b[index]=std::cos(float(r*columns+k)*.027f)*.2f;}''')
replace('out=[device newBufferWithLength:a.size()*4 options:MTLResourceStorageModeShared];','layout=[device newBufferWithLength:rows*stride*4 options:MTLResourceStorageModeShared],out=[device newBufferWithLength:rows*stride*4 options:MTLResourceStorageModeShared];')
replace('ref[4][2]','ref[4][3]')
replace('std::fill_n(static_cast<float*>(out.contents),a.size(),99.5f);','std::fill_n(static_cast<float*>(layout.contents),rows*stride,99.5f);std::fill_n(static_cast<float*>(out.contents),rows*stride,99.5f);')
replace('[e setBuffer:out offset:0 atIndex:4];[e setBytes:&norm length:sizeof(norm) atIndex:5];','[e setBuffer:layout offset:0 atIndex:4];[e setBuffer:out offset:0 atIndex:5];[e setBytes:&fp length:sizeof(fp) atIndex:6];')
replace('MTLSizeMake(rows*columns,1,1)','MTLSizeMake(batch*chunk*columns,1,1)')
replace('std::min(uint64_t(pipe[0].maxTotalThreadsPerThreadgroup),rows*columns)','std::min(uint64_t(pipe[0].maxTotalThreadsPerThreadgroup),batch*chunk*columns)')
replace('e=[command computeCommandEncoder];[e setComputePipelineState:pipe[1]];[e setBuffer:residual offset:0 atIndex:0];','''e=[command computeCommandEncoder];[e setComputePipelineState:pipe[3]];[e setBuffer:residual offset:0 atIndex:0];[e setBuffer:residual offset:0 atIndex:1];[e setBuffer:layout offset:0 atIndex:2];[e setBytes:&lp length:sizeof(lp) atIndex:3];[e dispatchThreads:MTLSizeMake(rows*columns,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(uint64_t(pipe[3].maxTotalThreadsPerThreadgroup),rows*columns),1,1)];[e endEncoding];
     e=[command computeCommandEncoder];[e setComputePipelineState:pipe[1]];[e setBuffer:layout offset:0 atIndex:0];''')
replace('need(std::memcmp(out.contents,ref[variant][1].data(),a.size()*4)==0,"Norm bits or padding changed");','need(std::memcmp(layout.contents,ref[variant][1].data(),rows*stride*4)==0,"Layout bits or padding changed");need(std::memcmp(out.contents,ref[variant][2].data(),rows*stride*4)==0,"Norm bits or padding changed");')
replace('for(int op=0;op<2;++op){ref[v][op].resize(a.size());std::memcpy(ref[v][op].data(),op?out.contents:residual.contents,a.size()*4);}','for(int op=0;op<3;++op){size_t count=op?rows*stride:a.size();ref[v][op].resize(count);std::memcpy(ref[v][op].data(),op==0?residual.contents:(op==1?layout.contents:out.contents),count*4);}')
replace('float expected=a[r*stride+k]*scales[v]+b[r*stride+k]*scales[(v+1)%4];need(ref[v][0][r*stride+k]==expected,"Independent residual mismatch");','uint64_t index=(r/chunk)*packedStride+(r%chunk)*columns+k;float expected=a[index]*scales[v]+b[index]*scales[(v+1)%4];need(ref[v][0][index]==expected&&ref[v][1][r*stride+k]==expected,"Independent packed/layout residual mismatch");')
replace('double(ref[v][0][r*stride+k])*inverse','double(ref[v][1][r*stride+k])*inverse')
replace('double actual=ref[v][1][r*stride+k]','double actual=ref[v][2][r*stride+k]')
replace('ref[v][0][r*stride+k]==99.5f&&ref[v][1][r*stride+k]==99.5f','ref[v][1][r*stride+k]==99.5f&&ref[v][2][r*stride+k]==99.5f')
replace('run(true,v,true);','for(uint64_t r=0;r<batch;++r)for(uint64_t k=chunk*columns;k<packedStride;++k)need(ref[v][0][r*packedStride+k]==99.5f,"Packed padding changed");run(true,v,true);')
replace('@"rows":@(rows),','@"rows":@(rows),@"batch":@(batch),@"chunk":@(chunk),',2)
replace('for(int round=0;round<3;++round)','for(int round=0;round<2;++round)')
replace('for(int round=0;round<2;++round)', 'const double warmMs=ms(warm,Clock::now());for(int round=0;round<2;++round)')
replace('@"common_warm_ms":@(ms(warm,Clock::now()))','@"common_warm_ms":@(warmMs)')
if len(sys.argv)==5:
 assert sys.argv[4]=='--extra-chunks'
 replace('chunk:{8,16,22,26,32,64,80}','chunk:{2,4,24,48}')
Path(sys.argv[3]).write_bytes(text.encode())
