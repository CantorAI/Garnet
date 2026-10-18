#include "metal_context.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("output prefix required");
 Dl_info image{};
 if(!dladdr(reinterpret_cast<const void*>(&Garnet::MetalContext::AvailableDeviceName),&image)||!image.dli_fname||std::filesystem::weakly_canonical(image.dli_fname)!=std::filesystem::weakly_canonical(LEGACY_LIBRARY_PATH))throw std::runtime_error("program probe bound another library");
 Garnet::MetalContext gpu;auto programs=gpu.ShaderProgramSources();
 if(programs.size()!=2||programs[0].empty())throw std::runtime_error("program source contract failed");
 for(size_t i=0;i<2;++i){std::ofstream out(std::string(argv[1])+(i?".prefill.metal":".main.metal"),std::ios::binary);out.write(programs[i].data(),programs[i].size());out.flush();if(!out)throw std::runtime_error("cannot save full program source");}
 std::cout<<"Actual constructor MAIN/prefill programs saved; loaded library path verified\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
