#include "metal_context.h"
#include <iostream>
#include <stdexcept>
int main(){try{Garnet::MetalContext gpu;auto before=Garnet::MetalContext::MemoryStats();if(!before.recommended||before.taskStatus||before.virtualBytes<before.resident)throw std::runtime_error("Memory metadata unavailable");
uint64_t retained=0,afterScope=0;
@autoreleasepool{{auto a=gpu.Allocate(1024,1024);gpu.Zero(a);auto alias=a;auto b=Garnet::MetalContext::MemoryStats();if(b.registry!=before.registry||b.allocated<before.allocated+4*1024*1024)throw std::runtime_error("Live allocation missing from snapshot");a=Garnet::MetalMatrix();auto c=Garnet::MetalContext::MemoryStats();if(c.allocated<b.allocated)throw std::runtime_error("Retained alias lost allocation");retained=c.allocated;}
afterScope=Garnet::MetalContext::MemoryStats().allocated;}
auto after=Garnet::MetalContext::MemoryStats();if(after.allocated>=retained)throw std::runtime_error("Explicit pool drain did not release allocation");std::cout<<"Device allocation/alias/release with explicit pool and task metadata PASS before="<<before.allocated<<" retained="<<retained<<" after_cpp_scope="<<afterScope<<" after_pool="<<after.allocated<<" recommended="<<before.recommended<<"\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
