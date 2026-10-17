#include "metal_context.h"
#include <iostream>
#include <stdexcept>
#include <limits>
int main(){try{Garnet::MetalContext gpu;for(auto dims:{std::pair<size_t,size_t>{1,1},{3,63},{2,257}}){auto bytes=gpu.MatrixStorageBytes(dims.first,dims.second);auto a=gpu.Allocate(dims.first,dims.second);auto info=Garnet::MetalContext::MatrixBufferInfo(a);if(bytes!=info.lengthBytes||bytes<dims.first*dims.second*4)throw std::runtime_error("Deferred storage descriptor differs from allocated layout");}
for(auto dims:{std::pair<size_t,size_t>{0,1},{1,0},{std::numeric_limits<size_t>::max(),2}}){bool rejected=false;try{gpu.MatrixStorageBytes(dims.first,dims.second);}catch(const std::exception&){rejected=true;}if(!rejected)throw std::runtime_error("Deferred storage admitted invalid geometry");}std::cout<<"Deferred descriptor bytes match MTLBuffer length; zero/overflow geometries rejected PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
