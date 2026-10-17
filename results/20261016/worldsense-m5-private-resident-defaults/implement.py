from pathlib import Path
import shutil
old=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-decode1024/source');new=old.parent.parent/'worldsense-private-resident-defaults/source';assert not new.exists();shutil.copytree(old,new)
p=new/'src/backends/metal/compiled_graph.h';s=p.read_text();s=s.replace('    bool BuildBF16Private(X::Value,X::Value,const std::string&);','    bool BuildBF16Private(X::Value,X::Value,const std::string&);\n    bool BuildBF16PrivateResident(X::Value,X::Value,const std::string&);');s=s.replace('        APISET().AddFunc<3>("build_bf16_private",&MetalCompiledGraph::BuildBF16Private);','        APISET().AddFunc<3>("build_bf16_private",&MetalCompiledGraph::BuildBF16Private);\n        APISET().AddFunc<3>("build_bf16_private_resident",&MetalCompiledGraph::BuildBF16PrivateResident);');p.write_text(s)
p=new/'src/backends/metal/compiled_graph.cpp';s=p.read_text();needle='    const bool result=BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,peer->bf16Projections_,residentDefaults&&!privateStorage);';assert needle in s;s=s.replace(needle,'''#ifdef GARNET_METAL_PRIVATE_RESIDENT_DEFAULTS
    const bool useResidentDefaults=residentDefaults;
#else
    const bool useResidentDefaults=residentDefaults&&!privateStorage;
#endif
    const bool result=BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,peer->bf16Projections_,useResidentDefaults);''')
s+='''\nnamespace Garnet {
bool MetalCompiledGraph::BuildBF16PrivateResident(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_PRIVATE_RESIDENT_DEFAULTS
    constexpr bool deferred=true;
#else
    constexpr bool deferred=false;
#endif
    const bool result=BuildInternal(std::move(graph),std::move(symbols),checkpoint,{},true,deferred);
    privateStorage_=result;return result;
}
}\n''';p.write_text(s)
p=new/'xModel/qwen/metal_dense/decoder.py';s=p.read_text();needle='            graph.build_bf16_private(output,symbols,checkpoint)';assert needle in s;s=s.replace(needle,'''            if self.resident_construction_requested and "private_resident_defaults" in garnet.backend_info():
                graph.build_bf16_private_resident(output,symbols,checkpoint)
            else:graph.build_bf16_private(output,symbols,checkpoint)''');p.write_text(s)
p=new/'src/entry/metal_main.cpp';s=p.read_text();needle='#ifdef GARNET_METAL_DENSE_PREFIX_CACHE';s=s.replace(needle,'#ifdef GARNET_METAL_PRIVATE_RESIDENT_DEFAULTS\n        info+="; private_resident_defaults";\n#endif\n'+needle,1);p.write_text(s)
p=new/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+'''\noption(GARNET_METAL_PRIVATE_RESIDENT_DEFAULTS "Explicit deferred default state storage for private resident graphs" OFF)
if(GARNET_METAL_PRIVATE_RESIDENT_DEFAULTS)
 if(NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS OR NOT GARNET_METAL_DENSE_PRIVATE_STORAGE OR NOT GARNET_METAL_PRIVATE_EXECUTION)
  message(FATAL_ERROR "Private resident defaults requires lazy resident, dense private, and private execution capabilities")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIVATE_RESIDENT_DEFAULTS=1)
endif()
''')
w=Path(__file__).resolve().parent;s=(w.parent/'worldsense-dense-decode1024/build.py').read_text().replace('worldsense-dense-decode1024','worldsense-private-resident-defaults');s=s.replace("'-DGARNET_METAL_DENSE_DECODE1024='+mode.upper(),","'-DGARNET_METAL_DENSE_DECODE1024=OFF','-DGARNET_METAL_PRIVATE_RESIDENT_DEFAULTS='+mode.upper(),");(w/'build.py').write_text(s)
print('WS30isolatedprototypewritten;unbuilt/untested;allwidekernelsOFFboth')
