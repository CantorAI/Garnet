#include "primary_block_shader.h"
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
#include "primary_vector_prefill_shader.h"
#endif
#include <fstream>
int main(int argc,char**argv){std::ofstream out(argv[1],std::ios::binary);out<<Garnet::kPrimaryBlockShader;
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
out<<Garnet::kPrimaryVectorPrefillShader;
#endif
return !out;}
