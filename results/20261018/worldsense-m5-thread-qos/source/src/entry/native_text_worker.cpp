// SPDX-License-Identifier: Apache-2.0
#include "native_text_worker.h"
#include "native_worker_support.h"
#include "nlohmann/json.hpp"
#include <spawn.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include <sstream>
extern char** environ;
namespace Garnet {
using J=nlohmann::json;namespace fs=std::filesystem;
namespace {
std::recursive_mutex globalGate;
uint64_t workerSequence=0;
X::Value Invoke(const X::Value& fn,const std::vector<X::Value>& args={}){X::Value result;if(!fn.Call(args,result)){const char* text=x3_runtime_last_error(fn.runtime());throw X::Error(text&&*text?text:"Native worker RPC failed");}return result;}
J ReadJSON(const fs::path& path){std::ifstream in(path);if(!in)throw std::runtime_error("Native worker identity/readiness file unavailable");return J::parse(in);}
std::string Birth(){std::random_device rng;std::ostringstream out;out<<std::hex;for(int i=0;i<8;++i)out<<rng();return out.str();}
uint64_t Add(uint64_t a,uint64_t b){if(b>UINT64_MAX-a)throw std::overflow_error("Native cohort memory counter overflow");return a+b;}
bool Eligible(const std::string& root){try{auto c=ReadJSON(fs::path(root)/"config.json");return !c.contains("vision_config")&&c.value("model_type",std::string())=="qwen3"&&c.value("hidden_size",0)==2048&&c.value("num_hidden_layers",0)==28;}catch(...){return false;}}
}
struct NativeTextWorker::Impl {
 X::Runtime runtime;X::Value peer;pid_t pid=0;std::string birth,endpoint;fs::path folder;bool healthy=false;
 std::map<std::string,bool> routes;J identity;uint64_t calls=0;
 static std::vector<Impl*>& Owners(){static std::vector<Impl*> list;return list;}
 explicit Impl(X3Runtime* value):runtime(value){Owners().push_back(this);}
 ~Impl(){Stop(false);auto& owners=Owners();owners.erase(std::remove(owners.begin(),owners.end(),this),owners.end());}
 X::Value Str(const std::string& value){return X::Value::String(runtime.host(),value);}
 bool Alive(){if(pid<=0)return false;int status=0;pid_t result;do{result=waitpid(pid,&status,WNOHANG);}while(result<0&&errno==EINTR);if(result==0)return true;if(result==pid||result==-1){pid=0;healthy=false;}return false;}
 J Info(){
  if(!Alive()||!healthy)throw std::runtime_error("Native RAW text worker exited; explicit reload required, inference not retried");
  try{auto info=J::parse(Invoke(peer["info_json"]).ToString());if(info.at("birth_id")!=birth||info.at("support").at("pid")!=pid||info.at("support").at("parent_pid")!=getpid()||!info.at("support").at("watchdog_active").get<bool>())throw std::runtime_error("Native text worker ownership changed");return info;}
  catch(...){healthy=false;throw;}
 }

 void Start(){
  if(pid>0){Info();return;}
  if(healthy)throw std::logic_error("Invalid native worker state");
  const fs::path raw=fs::weakly_canonical(GARNET_METAL_TEXT_WORKER_RAW_ROOT),source=GARNET_METAL_SOURCE_ROOT,exe=GARNET_METAL_TEXT_WORKER_EXECUTABLE;
  if(!fs::is_regular_file(raw/"libgarnet.dylib")||!fs::is_regular_file(exe))throw std::runtime_error("Native RAW worker build/executable missing");
  birth=Birth();folder=fs::temp_directory_path()/("garnet-native-worker-"+std::to_string(getpid())+"-"+birth);
  if(!fs::create_directory(folder))throw std::runtime_error("Native worker launch directory exists");chmod(folder.c_str(),0700);
  auto ready=folder/"ready.json";if(++workerSequence>=(uint64_t(1)<<20))throw std::runtime_error("Native worker session namespace exhausted");
  const int64_t logical=(int64_t(1)<<48)+int64_t(getpid())*(int64_t(1)<<20)+int64_t(workerSequence);endpoint="lrpc:"+std::to_string(logical);
#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_QOS
  const std::string qosMode="1";
#else
  const std::string qosMode="0";
#endif
  std::vector<std::string> args={exe.string(),"-B",(source/"runtime/native_text_worker.py").string(),std::to_string(logical),std::to_string(getpid()),birth,raw.string(),source.string(),ready.string(),qosMode};
  std::vector<char*> argv;for(auto& value:args)argv.push_back(value.data());argv.push_back(nullptr);
  std::vector<std::string> env;for(char** e=environ;e&&*e;++e)if(std::string(*e).rfind("GARNET_NATIVE_WORKER_PARENT=",0)!=0)env.emplace_back(*e);env.push_back("GARNET_NATIVE_WORKER_PARENT="+std::to_string(getpid()));std::vector<char*> envp;for(auto& value:env)envp.push_back(value.data());envp.push_back(nullptr);
  posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0);posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,(folder/"worker.log").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);posix_spawn_file_actions_adddup2(&actions,STDOUT_FILENO,STDERR_FILENO);
  posix_spawnattr_t attributes;posix_spawnattr_init(&attributes);
  const int attrStatus=posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT);
  if(attrStatus){posix_spawnattr_destroy(&attributes);posix_spawn_file_actions_destroy(&actions);throw std::runtime_error("Native worker close-on-exec setup failed");}
  const int status=posix_spawn(&pid,exe.c_str(),&actions,&attributes,argv.data(),envp.data());posix_spawnattr_destroy(&attributes);posix_spawn_file_actions_destroy(&actions);if(status){pid=0;throw std::runtime_error("Native worker spawn failed: "+std::to_string(status));}
  try {
   const auto deadline=Now()+30;while(!fs::exists(ready)){if(!Alive())throw std::runtime_error("Native worker startup failed; log: "+(folder/"worker.log").string());if(Now()>deadline)throw std::runtime_error("Native worker startup deadline; log: "+(folder/"worker.log").string());std::this_thread::sleep_for(std::chrono::milliseconds(10));}
   auto readyInfo=ReadJSON(ready);if(!Alive()||readyInfo.at("birth_id")!=birth||readyInfo.at("support").at("pid")!=pid)throw std::runtime_error("Native worker readiness identity mismatch");
   peer=runtime.ImportRemote("garnet_text_worker",endpoint.c_str());healthy=true;identity=Info();const auto support=identity.at("support");
   if(fs::weakly_canonical(support.at("library_path").get<std::string>())!=raw/"libgarnet.dylib"||fs::weakly_canonical(support.at("source_root").get<std::string>())!=fs::weakly_canonical(source))throw std::runtime_error("Native worker library/source mismatch");
   const double clock=support.at("steady_seconds");if(!std::isfinite(clock)||clock>Now()+.0001)throw std::runtime_error("Native worker clock invalid");
  }catch(...){Stop(false);throw;}
 }
 X::Value Call(const char* name,const std::vector<X::Value>& args={}){Info();try{return Invoke(peer[name],args);}catch(...){if(std::string(name)=="infer_json"||!Alive())healthy=false;throw;}}
 void Stop(bool graceful) noexcept {
  healthy=false;
  if(pid>0&&graceful){try{std::ofstream stop(folder/"stop.request");stop<<"stop\n";}catch(...){}}
  if(pid>0){
   if(graceful)for(int i=0;i<200&&Alive();++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
   if(pid>0&&Alive()){kill(pid,SIGTERM);for(int i=0;i<100&&Alive();++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));}
   if(pid>0&&Alive()){kill(pid,SIGKILL);int status=0;while(waitpid(pid,&status,0)<0&&errno==EINTR){}pid=0;}
  }
  // Release RPC values only after owned peer exit, so cleanup cannot wait on a
  // hung server even when the host's ordinary LRPC request timeout is disabled.
  peer=X::Value();
 }

 J Memory(){auto info=Info();auto value=J::parse(Call("memory_json").ToString());if(value.at("task_info_status")!=0)throw std::runtime_error("Child memory snapshot unavailable");value["pid"]=pid;value["birth_id"]=birth;value["snapshot_monotonic"]=Now();return value;}
};
std::recursive_mutex& NativeTextWorker::Gate(){return globalGate;}
double NativeTextWorker::Now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
NativeTextWorker::NativeTextWorker(X3Runtime* runtime):impl_(new Impl(runtime)){}
bool NativeTextWorker::UsesRuntime(X3Runtime* runtime)const{return impl_->runtime.get()==runtime;}
NativeTextWorker::~NativeTextWorker(){std::lock_guard<std::recursive_mutex> lock(Gate());impl_.reset();}
std::string NativeTextWorker::Serve(X::Value root,X::Value xmodel,X::Value cache,X::Value profile,X::Value model,const std::function<std::string()>& local){
 const auto name=model.ToString();auto found=impl_->routes.find(name);
 if(found!=impl_->routes.end()&&!found->second)return local();

 const bool remote=found!=impl_->routes.end()?found->second:Eligible(root.ToString());impl_->routes.emplace(name,remote);if(!remote)return local();if(!impl_->healthy||!impl_->Alive())impl_->Stop(false);impl_->Start();auto result=impl_->Call("serve_model",{root,xmodel,cache,profile,model}).ToString();
 return result;
}
std::string NativeTextWorker::Status(X::Value model,const std::function<std::string()>& local){auto i=impl_->routes.find(model.ToString());if(i==impl_->routes.end()||!i->second)return local();return impl_->Call("serve_status_json",{model}).ToString();}
std::string NativeTextWorker::Policy(X::Value enabled,const std::function<std::string()>& local){
 if(impl_->pid>0)impl_->Info();
 auto result=J::parse(local());
 if(impl_->pid>0){auto child=J::parse(impl_->Call("set_benchmark_prefill_policy",{enabled}).ToString());if(!result.value("supported",false)&&child.value("supported",false))result=child;}
 return result.dump();
}

std::string NativeTextWorker::Infer(X::Value prompt,X::Value image,X::Value limit,X::Value model,double start,const std::function<std::string()>& local,const std::function<std::string()>& parentMemory,const std::function<std::string()>& parentPrimary,const std::string& parentBackend){
 const auto callerQoS=J::parse(NativeWorkerSupport::ThreadQoS());
 const auto parentPrimaryBefore=J::parse(parentPrimary());
 const auto route=impl_->routes.find(model.ToString());const bool remote=route!=impl_->routes.end()&&route->second;
 // Match InferJsonLocal: non-string/non-binary image values mean no image.
 // In particular Galaxy uses a default-invalid X::Value, which is not an IPC value.
 X::Value forwardedImage=image;if(remote&&!image.IsString()&&!image.IsBin())forwardedImage=impl_->Str("");
 auto result=J::parse(remote?impl_->Call("infer_json",{prompt,forwardedImage,limit,model}).ToString():local());if(result.value("status",std::string())!="ok")return result.dump();
 const double first=result.at("native_first_token_monotonic"),modelStart=result.at("native_model_start_monotonic"),completed=Now();
 if(!std::isfinite(first)||!std::isfinite(modelStart)||modelStart<start-.0001||first<modelStart||first>completed+.0001)throw std::runtime_error("Native gateway first-token clock outside owning request");
 auto& timings=result.at("timings");J meta={{"route",remote?"native_lrpc_raw_text":"local"},{"parent_pid",getpid()},{"parent_backend",parentBackend},{"raw_model_ttft_seconds",result.at("ttft_seconds")},{"raw_model_total_seconds",result.at("total_seconds")},{"model_memory_after_decode",timings.at("memory_after_decode")},{"gateway_started_monotonic",start},{"model_started_monotonic",modelStart},{"first_token_monotonic",first}};
 const auto parentPrimaryAfter=J::parse(parentPrimary());J parentDelta=J::object();
 for(const char* key:{"decode_encoded","prefill_encoded","head_encoded","vector_decode_encoded","vector_head_encoded"}){const auto before=parentPrimaryBefore.at(key).get<uint64_t>(),after=parentPrimaryAfter.at(key).get<uint64_t>();if(after<before)throw std::runtime_error("Parent primary encoded counter decreased inside serialized request");parentDelta[key]=after-before;}
 meta["parent_primary_storage_before"]=parentPrimaryBefore;meta["parent_primary_storage_after"]=parentPrimaryAfter;meta["parent_primary_encoded_delta"]=parentDelta;
 meta["parent_thread_qos_requested"]=callerQoS;
 auto parent=J::parse(parentMemory());if(parent.at("task_info_status")!=0)throw std::runtime_error("Parent memory unavailable");parent["pid"]=getpid();parent["snapshot_monotonic"]=Now();J parts=J::array({parent});J total=parent;
 for(auto* owner:Impl::Owners())if(owner->pid>0||owner->healthy){auto sample=owner->Memory();if(sample.at("metal_registry_id")!=parent.at("metal_registry_id"))throw std::runtime_error("Cohort GPU device mismatch");parts.push_back(sample);for(const char* key:{"metal_allocated_bytes","task_physical_footprint_bytes","task_resident_bytes","task_compressed_bytes","task_internal_bytes","task_virtual_bytes"})total[key]=Add(total.at(key).get<uint64_t>(),sample.at(key).get<uint64_t>());}
 total["cohort_process_count"]=parts.size();total["process_samples"]=parts;total["scope"]="Controlled process-set sums per category; conservative task accounting may double count shared pages. Snapshots, not peaks, unique physical RAM or residency. One parent device recommendation, never summed.";
 meta["process_count"]=parts.size();meta["child"]=(impl_->pid>0)?impl_->Info():J(nullptr);meta["timing_scope"]="Actual existing first event minus gateway entry; total through routing/IPC/fresh process snapshots, before final response serialization. Raw model times retained; outer Skill includes response serialization.";timings["native_gateway"]=meta;timings["memory_after_decode"]=total;
 result["ttft_seconds"]=first-start;result["total_seconds"]=Now()-start;if(result.at("total_seconds").get<double>()<result.at("ttft_seconds").get<double>())throw std::runtime_error("Native gateway timing reversed");++impl_->calls;return result.dump();
}
std::string NativeTextWorker::State(X::Value model){auto i=impl_->routes.find(model.ToString());if(i==impl_->routes.end()||!i->second)throw std::invalid_argument("Model is not routed to native text worker");return impl_->Call("state_json",{model}).ToString();}
std::string NativeTextWorker::Close(X::Value model){auto i=impl_->routes.find(model.ToString());if(i==impl_->routes.end()||!i->second)throw std::invalid_argument("Model is not routed to native text worker");auto result=impl_->Call("close_model",{model}).ToString();impl_->routes.erase(i);return result;}
std::string NativeTextWorker::Info(){J list=J::array();for(auto* owner:Impl::Owners())if(owner->pid>0||owner->healthy)list.push_back(owner->Info());return J{{"enabled",true},{"parent_pid",getpid()},{"children",list},{"calls",impl_->calls}}.dump();}
void NativeTextWorker::Shutdown(){impl_->Stop(true);for(auto i=impl_->routes.begin();i!=impl_->routes.end();)if(i->second)i=impl_->routes.erase(i);else++i;}
}
