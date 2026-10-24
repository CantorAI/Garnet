// SPDX-License-Identifier: Apache-2.0
// Independent experiment. Does not change Garnet's production matmul path.
#include "modelopt_cublaslt.h"
#include "modelopt_nvfp4.cuh"
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>
using Json=nlohmann::json;
void CU(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
void BL(cublasStatus_t code){if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(code)));}
template<typename T> Json AlgorithmAttribute(const cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attr){
 T value{};size_t written=0;const auto status=cublasLtMatmulAlgoConfigGetAttribute(&algo,attr,&value,sizeof(value),&written);
 Json result={{"status",int(status)},{"bytes_written",written},{"expected_bytes",sizeof(value)}};
 if(status==CUBLAS_STATUS_SUCCESS){if(written!=sizeof(value))throw std::runtime_error("Unexpected algorithm attribute size");result["value"]=value;}
 return result;
}
Json AlgorithmConfiguration(const cublasLtMatmulAlgo_t& algo){
 return {{"id",AlgorithmAttribute<int32_t>(algo,CUBLASLT_ALGO_CONFIG_ID)},
 {"tile",AlgorithmAttribute<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_TILE_ID)},
 {"split_k",AlgorithmAttribute<int32_t>(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM)},
 {"reduction",AlgorithmAttribute<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)},
 {"cta_swizzling",AlgorithmAttribute<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)},
 {"custom_option",AlgorithmAttribute<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)},
 {"stages",AlgorithmAttribute<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID)},
 {"inner_shape",AlgorithmAttribute<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)},
 {"cluster_shape",AlgorithmAttribute<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)}};
}
struct Buffer{void* p=nullptr;explicit Buffer(size_t bytes){CU(cudaMalloc(&p,bytes));}~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;};
struct Stream{cudaStream_t p=nullptr;Stream(){CU(cudaStreamCreate(&p));}~Stream(){cudaStreamDestroy(p);}};
struct Handle{cublasLtHandle_t p=nullptr;Handle(){BL(cublasLtCreate(&p));}~Handle(){cublasLtDestroy(p);}};
struct Descriptors{
 cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;cublasLtMatmulPreference_t pref=nullptr;
 ~Descriptors(){if(pref)cublasLtMatmulPreferenceDestroy(pref);if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);if(a)cublasLtMatrixLayoutDestroy(a);if(op)cublasLtMatmulDescDestroy(op);}
};
__global__ void TransposeOutput(const __nv_bfloat16* source,__nv_bfloat16* output,int batch,int padded,int rows){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(batch)*rows)return;
 output[i]=source[(i%rows)*padded+i/rows];
}
__global__ void Evict(unsigned int* data,size_t count,unsigned int iteration){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)data[i]=data[i]+iteration+unsigned(i);
}
void Swapped(cublasLtHandle_t handle,const void* w,const void* ws,const void* x,const void* xs,void* tmp,void* out,
 int batch,int padded,int rows,int k,void* workspace,size_t workspaceBytes,cudaStream_t stream){
 Descriptors d;BL(cublasLtMatmulDescCreate(&d.op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
 const cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
 BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
 BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
 const cublasLtMatmulMatrixScale_t mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
 BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
 BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
 BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&xs,sizeof(xs)));
 BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&ws,sizeof(ws)));
 // Both paths use TN. Swap operands, pad activation rows with exact zeros,
 // then explicitly restore the row-major batch x channels output contract.
 BL(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_4F_E2M1,k,padded,k));
 BL(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_4F_E2M1,k,rows,k));
 BL(cublasLtMatrixLayoutCreate(&d.c,CUDA_R_16BF,padded,rows,padded));
 BL(cublasLtMatmulPreferenceCreate(&d.pref));
 BL(cublasLtMatmulPreferenceSetAttribute(d.pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspaceBytes,sizeof(workspaceBytes)));
 cublasLtMatmulHeuristicResult_t candidates[8]{};int count=0;
 BL(cublasLtMatmulAlgoGetHeuristic(handle,d.op,d.a,d.b,d.c,d.c,d.pref,8,candidates,&count));
 const float alpha=.5f,beta=0;bool success=false;
 for(int i=0;i<count;++i){
  if(candidates[i].state!=CUBLAS_STATUS_SUCCESS)continue;
  auto code=cublasLtMatmul(handle,d.op,&alpha,x,d.a,w,d.b,&beta,tmp,d.c,tmp,d.c,&candidates[i].algo,workspace,workspaceBytes,stream);
  if(code==CUBLAS_STATUS_SUCCESS){success=true;break;}if(code!=CUBLAS_STATUS_NOT_SUPPORTED)BL(code);
 }
 if(!success)throw std::runtime_error("No supported swapped TN candidate");
 TransposeOutput<<<unsigned((size_t(batch)*rows+255)/256),256,0,stream>>>(static_cast<const __nv_bfloat16*>(tmp),static_cast<__nv_bfloat16*>(out),batch,padded,rows);CU(cudaGetLastError());
}
// Prepare one original-layout candidate outside capture. Selection errors are
// retained separately; asynchronous CUDA or numerical failures remain fatal.
struct OriginalCandidate {
 Descriptors d;cublasLtMatmulHeuristicResult_t candidates[32]{};int count=0;
 OriginalCandidate(cublasLtHandle_t handle,const void* ws,const void* xs,int batch,int rows,int k,size_t bytes){
  BL(cublasLtMatmulDescCreate(&d.op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  const cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
  BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
  BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  const cublasLtMatmulMatrixScale_t mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
  BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
  BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&ws,sizeof(ws)));
  BL(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&xs,sizeof(xs)));
  BL(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_4F_E2M1,k,rows,k));
  BL(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_4F_E2M1,k,batch,k));
  BL(cublasLtMatrixLayoutCreate(&d.c,CUDA_R_16BF,rows,batch,rows));
  BL(cublasLtMatmulPreferenceCreate(&d.pref));
  BL(cublasLtMatmulPreferenceSetAttribute(d.pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
  BL(cublasLtMatmulAlgoGetHeuristic(handle,d.op,d.a,d.b,d.c,d.c,d.pref,32,candidates,&count));
 }
 cublasStatus_t Launch(int index,cublasLtHandle_t handle,const void* w,const void* x,void* out,void* workspace,size_t bytes,cudaStream_t stream){
  const float alpha=.5f,beta=0;
  return cublasLtMatmul(handle,d.op,&alpha,w,d.a,x,d.b,&beta,out,d.c,out,d.c,&candidates[index].algo,workspace,bytes,stream);
 }
};
struct TimedGraph{
 cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;cudaEvent_t start=nullptr,stop=nullptr;size_t nodes=0;
 ~TimedGraph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
 void Capture(cudaStream_t stream,const std::function<void()>& launch){
  CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  try{launch();}catch(...){cudaGraph_t discarded=nullptr;cudaStreamEndCapture(stream,&discarded);if(discarded)cudaGraphDestroy(discarded);throw;}
  CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphGetNodes(graph,nullptr,&nodes));if(!nodes)throw std::runtime_error("Empty captured matmul graph");
  std::vector<cudaGraphNode_t> all(nodes);CU(cudaGraphGetNodes(graph,all.data(),&nodes));size_t edges=0;CU(cudaGraphGetEdges(graph,nullptr,nullptr,nullptr,&edges));
  std::vector<cudaGraphNode_t> from(edges),to(edges);if(edges)CU(cudaGraphGetEdges(graph,from.data(),to.data(),nullptr,&edges));
  std::vector<cudaGraphNode_t> roots,leaves;
  for(auto node:all){if(std::find(to.begin(),to.end(),node)==to.end())roots.push_back(node);if(std::find(from.begin(),from.end(),node)==from.end())leaves.push_back(node);}
  CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));cudaGraphNode_t begin=nullptr,end=nullptr;
  CU(cudaGraphAddEventRecordNode(&begin,graph,nullptr,0,start));
  for(auto node:roots)CU(cudaGraphAddDependencies(graph,&begin,&node,nullptr,1));
  CU(cudaGraphAddEventRecordNode(&end,graph,leaves.data(),leaves.size(),stop));CU(cudaGraphInstantiate(&executable,graph,0));
 }
 float Run(cudaStream_t stream){CU(cudaGraphLaunch(executable,stream));CU(cudaStreamSynchronize(stream));float ms=0;CU(cudaEventElapsedTime(&ms,start,stop));return ms;}
};
size_t Offset(int row,int block,int paddedBlocks){return size_t(row/128)*128*paddedBlocks+size_t(block/4)*512+(row%32)*16+((row%128)/32)*4+block%4;}
void Write(const std::filesystem::path& path,const std::vector<__nv_bfloat16>& values){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*2);if(!f)throw std::runtime_error("Output write failed");}
Json Case(const std::filesystem::path& out,cublasLtHandle_t handle,cudaStream_t stream,Buffer& flush,size_t flushBytes,int rows,int k,int batch,bool timing,bool sweep=false){
 const int padded=std::max(16,((batch+15)/16)*16),blocks=k/16,paddedBlocks=((blocks+3)/4)*4;
 std::vector<unsigned char> w(size_t(rows)*k/2,0x22),x(size_t(padded)*k/2,0),ws(Garnet::ModelOptNVFP4ScaleBytes(rows,k),0),xs(Garnet::ModelOptNVFP4ScaleBytes(padded,k),0);
 std::fill_n(x.begin(),size_t(batch)*k/2,0x22);const unsigned char codes[]={56,64,72};const int scales[]={1,2,4};
 for(int r=0;r<rows;++r)for(int b=0;b<blocks;++b)ws[Offset(r,b,paddedBlocks)]=codes[(r+b)%3];
 for(int r=0;r<batch;++r)for(int b=0;b<blocks;++b)xs[Offset(r,b,paddedBlocks)]=codes[(r+b+1)%3];
 Buffer dw(w.size()),dx(x.size()),dws(ws.size()),dxs(xs.size()),a(size_t(batch)*rows*2),b(size_t(batch)*rows*2),tmp(size_t(padded)*rows*2),workspace((sweep?64ULL:4ULL)<<20);
 CU(cudaMemcpyAsync(dw.p,w.data(),w.size(),cudaMemcpyHostToDevice,stream));CU(cudaMemcpyAsync(dx.p,x.data(),x.size(),cudaMemcpyHostToDevice,stream));
 CU(cudaMemcpyAsync(dws.p,ws.data(),ws.size(),cudaMemcpyHostToDevice,stream));CU(cudaMemcpyAsync(dxs.p,xs.data(),xs.size(),cudaMemcpyHostToDevice,stream));
 auto original=[&]{BL(Garnet::RunModelOptNVFP4Matmul(handle,dw.p,dws.p,dx.p,dxs.p,a.p,batch,rows,k,.5f,workspace.p,4ULL<<20,stream));};
 auto swapped=[&]{Swapped(handle,dw.p,dws.p,dx.p,dxs.p,tmp.p,b.p,batch,padded,rows,k,workspace.p,4ULL<<20,stream);};
 for(int i=0;i<2;++i){original();swapped();}CU(cudaStreamSynchronize(stream));
 std::vector<__nv_bfloat16> left(size_t(batch)*rows),right(left.size());CU(cudaMemcpy(left.data(),a.p,left.size()*2,cudaMemcpyDeviceToHost));CU(cudaMemcpy(right.data(),b.p,right.size()*2,cudaMemcpyDeviceToHost));
 const auto stem="n"+std::to_string(rows)+"-k"+std::to_string(k)+"-b"+std::to_string(batch);
 Write(out/(stem+"-original.bf16"),left);Write(out/(stem+"-swapped.bf16"),right);
 for(int batchRow=0;batchRow<batch;++batchRow)for(int row=0;row<rows;++row){
  int period=0,remainder=0;for(int j=0;j<3;++j)period+=scales[(row+j)%3]*scales[(batchRow+j+1)%3];
  for(int j=0;j<blocks%3;++j)remainder+=scales[(row+j)%3]*scales[(batchRow+j+1)%3];
  float expected=__bfloat162float(__float2bfloat16(float(8*(period*(blocks/3)+remainder))));const auto i=size_t(batchRow)*rows+row;
  if(__bfloat162float(left[i])!=expected||__bfloat162float(right[i])!=expected)throw std::runtime_error("Independent scale/transpose oracle mismatch: "+stem);
 }
 Json result={{"stem",stem},{"batch",batch},{"swapped_padded_rows",padded},{"rows",rows},{"columns",k},{"outputs_checked",left.size()},{"exact_independent_oracle",true},{"workspace_bytes",4ULL<<20}};
 if(timing){
  TimedGraph normal,alternate;normal.Capture(stream,original);alternate.Capture(stream,swapped);normal.Run(stream);alternate.Run(stream);
  Json trials=Json::array();for(int repeat=0;repeat<4;++repeat)for(int mode:{0,1,1,0}){
   Evict<<<1024,256,0,stream>>>(static_cast<unsigned int*>(flush.p),flushBytes/4,unsigned(repeat+mode+1));CU(cudaGetLastError());
   const float ms=(mode?alternate:normal).Run(stream);if(!(ms>0))throw std::runtime_error("Invalid CUDA event timing");
   trials.push_back({{"mode",mode?"swapped":"original"},{"abba_repeat",repeat},{"milliseconds",ms}});
  }
  result["trials"]=trials;result["captured_nodes_before_timer_nodes"]={normal.nodes,alternate.nodes};
 }
 if(sweep){
  result["algorithm_sweep"]=Json::array();
  for(size_t budget:{4ULL<<20,32ULL<<20,64ULL<<20}){
   OriginalCandidate options(handle,dws.p,dxs.p,batch,rows,k,budget);
   Json group={{"workspace_budget_bytes",budget},{"heuristic_requested",32},{"heuristic_returned",options.count},{"candidates",Json::array()}};
   for(int index=0;index<options.count;++index){
    const auto& candidate=options.candidates[index];
    Json row={{"heuristic_index",index},{"heuristic_state",int(candidate.state)},{"workspace_required_bytes",candidate.workspaceSize}};
    row["algorithm_configuration"]=AlgorithmConfiguration(candidate.algo);
    row["estimated_waves"]=candidate.wavesCount;
    int algorithmId=-1;size_t written=0;
    const auto idStatus=cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_ID,&algorithmId,sizeof(algorithmId),&written);
    row["algorithm_id_status"]=int(idStatus);if(idStatus==CUBLAS_STATUS_SUCCESS)row["algorithm_id"]=algorithmId;
    if(candidate.state!=CUBLAS_STATUS_SUCCESS){row["status"]="heuristic_rejected";group["candidates"].push_back(row);continue;}
    auto launch=[&]{return options.Launch(index,handle,dw.p,dx.p,a.p,workspace.p,budget,stream);};
    const auto status=launch();row["launch_status"]=int(status);
    if(status==CUBLAS_STATUS_NOT_SUPPORTED){row["status"]="not_supported";group["candidates"].push_back(row);continue;}
    BL(status);CU(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> actual(left.size());CU(cudaMemcpy(actual.data(),a.p,actual.size()*2,cudaMemcpyDeviceToHost));
    if(std::memcmp(actual.data(),left.data(),left.size()*2)){Write(out/(stem+"-candidate-"+std::to_string(index)+"-workspace-"+std::to_string(budget)+"-mismatch.bf16"),actual);throw std::runtime_error("Algorithm candidate differs from independent structured oracle: "+stem+" index="+std::to_string(index)+" workspace="+std::to_string(budget));}
    TimedGraph selected,baseline;selected.Capture(stream,[&]{BL(launch());});baseline.Capture(stream,original);
    selected.Run(stream);baseline.Run(stream);Json trials=Json::array();
    for(int repeat=0;repeat<4;++repeat)for(int mode:{0,1,1,0}){
     Evict<<<1024,256,0,stream>>>(static_cast<unsigned int*>(flush.p),flushBytes/4,unsigned(repeat+mode+1));CU(cudaGetLastError());
     const float ms=(mode?selected:baseline).Run(stream);if(!(ms>0))throw std::runtime_error("Invalid algorithm event timing");
     trials.push_back({{"mode",mode?"candidate":"baseline_first_supported"},{"abba_repeat",repeat},{"milliseconds",ms}});
    }
    row["status"]="structured_oracle_passed";row["outputs_checked"]=left.size();row["trials"]=trials;
    group["candidates"].push_back(row);
   }
   result["algorithm_sweep"].push_back(group);
  }
 }
 return result;
}
int main(int argc,char** argv){Json report={{"status","starting"},{"cases",Json::array()},{"scope","Standalone synthetic packed NVFP4/scales oracle. Actual model projection dimensions when requested, not checkpoint weights or model tokens. Timings use CUDA event nodes inside captured graphs, include transpose/padded-row cost, exclude preceding cache-eviction kernel. Eviction is cold-cache intent, not proof of exact hardware cache residency. No full-model, HTTP or2x claim."}};std::filesystem::path path;
 try{
  if(argc<2||argc>3)throw std::runtime_error("Pass fixed report path and optional --shape-timing or --algorithm-timing");path=argv[1];if(std::filesystem::exists(path))throw std::runtime_error("Preserve prior probe report");std::filesystem::create_directories(path.parent_path());
  const bool sweep=argc==3&&std::string(argv[2])=="--algorithm-timing";const bool timing=sweep||(argc==3&&std::string(argv[2])=="--shape-timing");if(argc==3&&!timing)throw std::runtime_error("Unknown argument");
  cudaDeviceProp prop{};CU(cudaGetDeviceProperties(&prop,0));if(prop.major<10)throw std::runtime_error("Actual Blackwell GPU required");
  int runtime=0;CU(cudaRuntimeGetVersion(&runtime));report["device"]={{"name",prop.name},{"major",prop.major},{"minor",prop.minor},{"l2_cache_bytes",prop.l2CacheSize}};report["cuda_runtime_version"]=runtime;report["cublaslt_version_raw"]=cublasLtGetVersion();
  const size_t flushBytes=std::max<size_t>(256ULL<<20,2ULL*prop.l2CacheSize+(32ULL<<20));report["flush_bytes"]=flushBytes;
  Stream stream;Handle handle;Buffer flush(flushBytes);CU(cudaMemsetAsync(flush.p,0,flushBytes,stream.p));
  auto save=[&]{std::ofstream file(path);file<<report.dump(2);};save();
  for(int batch:{1,4,128}){report["cases"].push_back(Case(path.parent_path(),handle.p,stream.p,flush,flushBytes,256,256,batch,false));save();}
  if(timing)for(auto shape:std::vector<std::pair<int,int>>{{17408,5120},{34816,5120},{5120,17408}})for(int batch:{1,4,128}){report["cases"].push_back(Case(path.parent_path(),handle.p,stream.p,flush,flushBytes,shape.first,shape.second,batch,true,sweep));save();}
  report["status"]="passed";save();std::cout<<"NVFP4_OPERAND_SWAP_ORACLE_PASS; component timing only if requested\n";return 0;
 }catch(const std::exception& error){report["status"]="failed";report["error"]=error.what();if(!path.empty()&&!std::filesystem::exists(path)){std::ofstream file(path);file<<report.dump(2);}else if(!path.empty()&&report.contains("device")){std::ofstream file(path);file<<report.dump(2);}std::cerr<<error.what()<<'\n';return 1;}}
