#include "xlang3/xlang3.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
using J=nlohmann::json;
static X::Value call(const X::Value& fn){X::Value out;if(!fn.Call({},out)){const char*e=x3_runtime_last_error(fn.runtime());throw std::runtime_error(e&&*e?e:"native RPC failed");}return out;}
int main(int argc,char**argv){try{if(argc!=5)throw std::runtime_error("endpoint,pid,birth,output required");X::Runtime runtime;auto peer=runtime.ImportRemote("qos_context_probe",argv[1]);J data;data["identity"]=J::parse(call(peer["identity"]).ToString());if(data["identity"]["pid"]!=std::stoll(argv[2])||data["identity"]["birth"]!=argv[3])throw std::runtime_error("QoS owned peer identity mismatch");data["rpc_requested_context"]=J::parse(call(peer["report"]).ToString());data["rpc_scope_probe"]=J::parse(call(peer["probe"]).ToString());data["rpc_recovery_after_scope"]=J::parse(call(peer["report"]).ToString());if(!data["rpc_scope_probe"]["native_API_scope_contract_pass"].get<bool>()||data["rpc_recovery_after_scope"]["active_owned_scopes"]!=0)throw std::runtime_error("QoS RPC scope recovery failed");if(!call(peer["stop"]).ToLongLong())throw std::runtime_error("Owned peer shutdown failed");std::ofstream(argv[4])<<data.dump(2)<<'\n';std::cout<<"Native XLang3 LRPC requested-QoS/owned nested/end/unwind scope contract PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
