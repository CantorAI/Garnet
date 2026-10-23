// SPDX-License-Identifier: Apache-2.0
#include "../../src/backends/tensorrt/bf16_projection_plugin.h"
#include <NvInfer.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(){try{
    require(Garnet::EnsureBF16ProjectionPluginRegistered(),"Registry failed");
    for(bool workspace:{false,true}){
        const char* version=workspace?"2":"1";
        auto* creator=getPluginRegistry()->getPluginCreator("GarnetBF16Projection",version,"");
        require(creator!=nullptr,"Version missing");
        uint32_t bytes=16384,invalid=32768;
        require(creator->deserializePlugin("wrong-size",&bytes,1)==nullptr,"Invalid size accepted");
        if(workspace){
            require(creator->deserializePlugin("null",nullptr,4)==nullptr,"Null data accepted");
            require(creator->deserializePlugin("unsupported",&invalid,4)==nullptr,"Unsupported scratch accepted");
        }
        auto* plugin=creator->deserializePlugin("valid",workspace?&bytes:nullptr,workspace?4:0);
        require(plugin!=nullptr,"Valid payload rejected");plugin->setPluginNamespace("contract");
        auto* clone=dynamic_cast<nvinfer1::IPluginV2DynamicExt*>(plugin->clone());require(clone!=nullptr,"Clone failed");
        require(!std::strcmp(clone->getPluginNamespace(),"contract"),"Clone namespace mismatch");
        require(!std::strcmp(clone->getPluginVersion(),version),"Clone version mismatch");
        require(clone->getSerializationSize()==(workspace?4:0),"Size mismatch");
        require(clone->getWorkspaceSize(nullptr,0,nullptr,0)==(workspace?16384:0),"Workspace mismatch");
        uint32_t restored=0;clone->serialize(&restored);require(restored==(workspace?16384:0),"Payload mismatch");
        clone->destroy();plugin->destroy();
    }
    std::cout<<"BF16_PROJECTION_PLUGIN_CONTRACT_COMPLETE versions 2\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
