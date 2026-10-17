from pathlib import Path
import shutil,difflib,json
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/out/research');old=r/'worldsense-weight-residency/source';oldfault=old.parent/'fault-source';source=r/'worldsense-request-residency/source';target=source.parent/'fault-source';assert not target.exists();shutil.copytree(source,target)
for rel in ['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h']:
 a=(old/rel).read_text().splitlines(True);b=(oldfault/rel).read_text().splitlines(True);text=(source/rel).read_text()
 for group in list(difflib.SequenceMatcher(None,a,b,autojunk=False).get_grouped_opcodes(3))[::-1]:
  aa=''.join(a[group[0][1]:group[-1][2]]);bb=''.join(b[group[0][3]:group[-1][4]]);assert text.count(aa)==1,rel;text=text.replace(aa,bb,1)
 (target/rel).write_text(text)
rel='test2026/metal/weight_residency_test.cpp';assert (old/rel).read_bytes()==(source/rel).read_bytes();shutil.copy2(oldfault/rel,target/rel)
p=target/'src/backends/metal/metal_context.mm';s=p.read_text();needle='WeightResidencyBudget& ResidencyBudget(){static WeightResidencyBudget b(kWeightResidencyLimit);return b;}';assert needle in s;s=s.replace(needle,'''WeightResidencyBudget& ResidencyBudget(){
#ifdef GARNET_METAL_WEIGHT_RESIDENCY_TEST_STATE
 static WeightResidencyBudget b(65536);
#else
 static WeightResidencyBudget b(kWeightResidencyLimit);
#endif
 return b;}
''')
start=s.index('std::string MetalContext::WeightResidencyStatsJSON() const {');end=s.index('\n}\n}',start);part=s[start:end];needle=' return result;';assert needle in part;part=part.replace(needle,'''#if defined(GARNET_METAL_WEIGHT_RESIDENCY_TEST_STATE) && defined(GARNET_METAL_WEIGHT_RESIDENCY)
 uint64_t count=0;if(@available(macOS 15.0,*)){if(impl_->weightResidencySet)count=impl_->weightResidencySet.allocationCount;}
 result.pop_back();result+=",\\"test_owners\\":"+std::to_string(impl_->weightResidencyOwners.size())+",\\"test_set_count\\":"+std::to_string(count)+",\\"test_attached\\":"+std::to_string(impl_->weightResidencyAttached)+",\\"test_requested\\":"+std::to_string(impl_->weightResidencyRequested)+",\\"test_global_used\\":"+std::to_string(ResidencyBudget().Used(impl_->device.registryID))+"}";
#endif
'''+needle);p.write_text(s[:start]+part+s[end:])
p=target/'src/backends/metal/decoder_session.h';s=p.read_text();needle='    bool BeginWeightResidencyRequest();';assert needle in s;s=s.replace(needle,'''#ifdef GARNET_METAL_WEIGHT_RESIDENCY_TEST_STATE
    bool TestRequestThreadOwnership();
#endif
'''+needle);needle='        APISET().AddFunc<0>("begin_weight_residency_request",&MetalDecoderSession::BeginWeightResidencyRequest);';assert needle in s;s=s.replace(needle,'''#ifdef GARNET_METAL_WEIGHT_RESIDENCY_TEST_STATE
        APISET().AddFunc<0>("test_request_thread_ownership",&MetalDecoderSession::TestRequestThreadOwnership);
#endif
'''+needle);p.write_text(s)
p=target/'src/backends/metal/decoder_session.cpp';s=p.read_text();s='#include <thread>\n'+s;s+='''
#ifdef GARNET_METAL_WEIGHT_RESIDENCY_TEST_STATE
namespace Garnet {
bool MetalDecoderSession::TestRequestThreadOwnership(){
 {std::lock_guard<std::mutex> lock(mutex_);if(!gpu_)throw std::logic_error("Closed test session");gpu_->CheckWeightResidencyRequestOwner();if(gpu_->WeightResidencyStatsJSON().find("\\"request_active\\":1")==std::string::npos)throw std::logic_error("Test requires active request");}
 int passed=0;std::thread worker([&]{for(int i=0;i<7;i++){try{switch(i){case 0:Reset();break;case 1:CapturePrefix(64);break;case 2:RestorePrefix(1);break;case 3:ClearPrefixCache();break;case 4:EndWeightResidencyRequest();break;case 5:BeginWeightResidencyRequest();break;case 6:Run(X::Value(),X::Value(),X::Value(),X::Value(),X::Value(),-1);break;}}catch(const std::exception& e){auto m=std::string(e.what());if(m.find("another thread")!=std::string::npos||(i==5&&m.find("Nested")!=std::string::npos))++passed;}}});worker.join();return passed==7;
}
}
#endif
''';p.write_text(s)
changed=[str(p.relative_to(target))for p in target.rglob('*')if p.is_file()and'__pycache__'not in p.parts and (not(source/p.relative_to(target)).exists()or p.read_bytes()!=(source/p.relative_to(target)).read_bytes())];assert len(changed)==7,changed
(w/'fault-preparation-notes.json').write_text(json.dumps({'changed_files':sorted(changed),'test_global_budget_bytes':65536,'measured_global_budget_bytes':4294967296,'test_only_cross_thread_methods':7,'new_request_scope_source_preserved':True,'not_built_yet':True},indent=2));print('Separate7filefaultdelta prepared;no measuredsource mutation')
