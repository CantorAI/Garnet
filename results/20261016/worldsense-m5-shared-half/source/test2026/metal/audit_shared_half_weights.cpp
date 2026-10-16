// Checkpoint representation/admission audit. No inference-speed measurement.
#include "metal_context.h"
#include "safetensors_index.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){try{
 if(argc!=3)return 2;Garnet::SafeTensorsIndex index;std::string error;if(!index.Open(argv[1],error))throw std::runtime_error(error);
 Garnet::MetalContext gpu;std::vector<std::string> names;for(const auto& p:index.Entries())if(p.first.find("model.language_model.layers.")==0&&p.second.dataType=="BF16"&&p.second.shape.size()==2)names.push_back(p.first);std::sort(names.begin(),names.end());
 nlohmann::json rows=nlohmann::json::array();uint64_t original=0,retained=0,admitted=0;size_t count=0;
 for(const auto& name:names){const auto& m=*index.Find(name);Garnet::SafeTensorsMappedFile file;if(!file.Open(m.filePath,error))throw std::runtime_error(error);auto* data=file.DataAt(m.dataOffset,m.dataSize);if(!data)throw std::runtime_error("Checkpoint tensor outside mapping");auto weight=gpu.UploadBF16(data,m.dataSize,m.shape[0],m.shape[1]);auto storage=weight.RetainedStorageBytes();original+=m.dataSize;retained+=storage;if(weight.SharedExactHalfReady()){admitted+=m.dataSize;++count;if(storage>m.dataSize+uint64_t(m.shape[0])*260)throw std::runtime_error("Second full weight buffer retained");}
 rows.push_back({{"name",name},{"shape",m.shape},{"original_bytes",m.dataSize},{"retained_storage_bytes",storage},{"shared_exact_half",weight.SharedExactHalfReady()},{"common_scale",weight.GlobalHalfReady()}});
 }
 if(names.empty())throw std::runtime_error("No model layer matrices found");nlohmann::json report={{"scope","Individual decoded-layer BF16 matrices; logical Metal buffer bytes, not model peak memory or speed"},{"matrices",rows},{"original_bytes",original},{"retained_bytes",retained},{"shared_original_bytes",admitted},{"shared_matrix_count",count}};std::ofstream out(argv[2]);out<<report.dump(2);if(!out)throw std::runtime_error("Cannot write audit");std::cout<<"Checkpoint shared storage audit complete: "<<count<<"/"<<names.size()<<" matrices\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
