#include "metal_context.h"
#include <vector>
#include <iostream>
using namespace Garnet;
int main(){try{MetalContext context;std::vector<uint16_t> bits(260*128,0x3f80);auto w=context.UploadBF16(bits.data(),bits.size()*2,260,128,false,false,true);uint64_t n=0;auto info=MetalContext::WeightBufferInfo(w);for(auto row:info){n+=row.allocatedBytes;std::cout<<row.representation<<" length="<<row.lengthBytes<<" allocated="<<row.allocatedBytes<<"\n";}std::cout<<"primary="<<w.PrimaryReady()<<" allocations="<<info.size()<<" sum="<<n<<"\n";return w.PrimaryReady()&&info.size()==2&&n<=65536?0:2;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
