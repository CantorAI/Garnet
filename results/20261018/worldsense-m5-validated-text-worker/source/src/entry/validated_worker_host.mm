// Native identity/clock/OS snapshots only. No model or image computation.
#include "xlang3/xlang3.h"
#include "native_worker_support.h"
#include "nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <dlfcn.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <array>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
namespace {
using J=nlohmann::json;namespace fs=std::filesystem;
constexpr const char* ExpectedSHA="dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72";
fs::path LegacyPath(){return fs::weakly_canonical(WORKER_LEGACY_LIBRARY);}
std::string FileSHA(const fs::path& path){
 std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Pinned native backend cannot be read");
 CC_SHA256_CTX state;CC_SHA256_Init(&state);std::array<char,65536> block{};
 while(input){input.read(block.data(),block.size());auto n=input.gcount();if(n>0)CC_SHA256_Update(&state,block.data(),static_cast<CC_LONG>(n));}
 if(!input.eof())throw std::runtime_error("Pinned native backend read failed");
 unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&state);
 std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto value:digest)out<<std::setw(2)<<unsigned(value);return out.str();
}
J LegacyIdentity(){
 const auto expected=LegacyPath();size_t count=0;
 for(uint32_t i=0;i<_dyld_image_count();++i){const char* name=_dyld_get_image_name(i);if(!name)continue;fs::path p=name;
  if(p.filename()=="libgarnet.dylib"){++count;if(fs::weakly_canonical(p)!=expected)throw std::runtime_error("A different native Garnet backend is loaded");}
 }
 if(count!=1)throw std::runtime_error("Exactly one pinned Garnet native backend must be loaded");
 void* handle=dlopen(expected.c_str(),RTLD_LAZY|RTLD_NOLOAD);if(!handle)throw std::runtime_error("Pinned Garnet backend is not loaded");
 J result;
 try{
  void* load=dlsym(handle,"Load");auto* abi=static_cast<const uint32_t*>(dlsym(handle,"xlang3_package_abi_version"));Dl_info image{};
  if(!load||!abi||*abi!=X3_ABI_VERSION||!dladdr(load,&image)||!image.dli_fname||fs::weakly_canonical(image.dli_fname)!=expected)throw std::runtime_error("Pinned native package symbol/ABI identity mismatch");
  static std::once_flag once;static std::string digest;
  std::call_once(once,[&]{auto actual=FileSHA(expected);if(actual!=ExpectedSHA)throw std::runtime_error("Pinned native backend SHA mismatch");digest=actual;});
  result={{"library_path",expected.string()},{"library_sha256",digest},{"package_abi",*abi},{"loaded_garnet_images",count},{"expected_original_source_root",WORKER_LEGACY_SOURCE},{"scope","Unique loaded image and Load/ABI address checked on each query; immutable binary SHA validated once before model startup. Separate before/after source/library hashes remain required."}};
 }catch(...){dlclose(handle);throw;}
 dlclose(handle);return result;
}
class WorkerHost {
public:
 std::string Support(){return Garnet::NativeWorkerSupport::Info(WORKER_HOST_ROOT);}
 std::string Legacy(){return LegacyIdentity().dump();}
 bool Watch(int64_t parent){LegacyIdentity();return Garnet::NativeWorkerSupport::WatchParent(parent);}
 double Monotonic(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
 std::string Memory(){
  LegacyIdentity();@autoreleasepool {
   auto device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("Native Metal memory device unavailable");
   task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
   auto status=task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count);
   return J{{"metal_allocated_bytes",device.currentAllocatedSize},{"metal_recommended_working_set_bytes",device.recommendedMaxWorkingSetSize},{"metal_registry_id",device.registryID},{"task_info_status",status},{"task_physical_footprint_bytes",info.phys_footprint},{"task_resident_bytes",info.resident_size},{"task_compressed_bytes",info.compressed},{"task_internal_bytes",info.internal},{"task_virtual_bytes",info.virtual_size},{"scope","Actual native current-process Metal and Mach task snapshot; not peak/free/unique physical RAM. Failure status is never converted to success."}}.dump();
  }
 }
 std::string RawPrimary(){
  LegacyIdentity();return J{{"active_weights",0},{"original_bytes",0},{"payload_bytes",0},{"metadata_bytes",0},{"allocated_bytes",0},{"uploads",0},{"decode_encoded",0},{"prefill_encoded",0},{"head_encoded",0},{"vector_decode_encoded",0},{"vector_head_encoded",0}}.dump();
 }
 std::string RawPolicies(){
  LegacyIdentity();return J{{"lazy_default_planned_slots",0},{"lazy_default_retired_bytes",0},{"lazy_default_rehydrated_bytes",0},{"counter_origin","Structural zeros only for packed-primary and lazy-retirement domains absent from the exact pinned WS10 implementation; not zero raw weights or invented OS counters."}}.dump();
 }
 BEGIN_PACKAGE(WorkerHost)
  APISET().AddFunc<0>("support_json",&WorkerHost::Support);
  APISET().AddFunc<0>("legacy_json",&WorkerHost::Legacy);
  APISET().AddFunc<1>("watch_parent",&WorkerHost::Watch);
  APISET().AddFunc<0>("monotonic",&WorkerHost::Monotonic);
  APISET().AddFunc<0>("memory_stats_json",&WorkerHost::Memory);
  APISET().AddFunc<0>("raw_primary_stats_json",&WorkerHost::RawPrimary);
  APISET().AddFunc<0>("raw_policies_json",&WorkerHost::RawPolicies);
 END_PACKAGE
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version=X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* rawHost,X3Value current){
 auto* host=static_cast<X3PackageHost*>(rawHost);if(!host||host->abi_version!=X3_ABI_VERSION)return X3_STATUS_ERROR;
 try{WorkerHost::BuildAPI();return WorkerHost::APISET().Create(host,"garnet_worker_host",current);}catch(...){return X3_STATUS_ERROR;}
}
