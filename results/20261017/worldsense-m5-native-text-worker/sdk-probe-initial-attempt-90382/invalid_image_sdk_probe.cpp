#include "xlang3/xlang3.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using J=nlohmann::json;
static X::Value call(const X::Value& fn,const std::vector<X::Value>& args={}){X::Value out;if(!fn.Call(args,out)){const char* e=x3_runtime_last_error(fn.runtime());throw std::runtime_error(e&&*e?e:"Native call failed");}return out;}
int main(int argc,char**argv){J record;std::string output;try{
 if(argc!=4)throw std::runtime_error("mode,expected_marshal_failure,output required");const std::string mode=argv[1];const bool expectFailure=std::string(argv[2])=="1";output=argv[3];X::Runtime runtime;const std::string s="/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source",w="/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker";runtime.AddImportRoot("/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-"+mode);runtime.AddImportRoot(s);X::Module garnet(runtime,"garnet");auto str=[&](const std::string& x){return X::Value::String(runtime.host(),x);};
 record["mode"]=mode;record["native_support"]=J::parse(call(garnet["native_worker_support_json"]).ToString());record["backend"]=call(garnet["backend_info"]).ToString();record["serve"]=J::parse(call(garnet["serve_model"],{str(w+"/invalid-text-fixture"),str(s+"/xModel/qwen/metal_dense"),str(w+"/invalid-text-cache"),str("{\"maxInputTokens\":64,\"maxOutputTokens\":8,\"textPrefixCache\":false}"),str("invalid-text")}).ToString());if(record["serve"]["ready"]!=false)throw std::runtime_error("Fixture unexpectedly loaded model");
 X::Value absentImage;record["default_image_is_valid"]=absentImage.IsValid();record["default_image_is_string"]=absentImage.IsString();record["default_image_is_binary"]=absentImage.IsBin();bool marshalFailure=false;
 try{record["call"]=J::parse(call(garnet["infer_json"],{str("test"),absentImage,X::Value(2),str("invalid-text")}).ToString());}catch(const std::exception& e){record["exception"]=e.what();marshalFailure=std::string(e.what()).find("XLangStream IPC marshal does not support this value yet")!=std::string::npos;if(!marshalFailure)throw;}
 record["marshal_failure"]=marshalFailure;if(marshalFailure!=expectFailure)throw std::runtime_error("Unexpected invalid-image boundary outcome");if(!marshalFailure&&record["call"]["status"]!="error")throw std::runtime_error("Missing model should return original structured error");
 if(mode!="off"&&!marshalFailure){record["info_after"]=J::parse(call(garnet["native_text_worker_info_json"],{str("")}).ToString());if(record["info_after"]["children"][0]["infer_calls"]!=1)throw std::runtime_error("Missing-model inference was retried");}
 call(garnet["shutdown"]);record["expected_boundary_outcome_pass"]=true;std::ofstream(output)<<record.dump(2)<<'\n';std::cout<<"Native SDK default-invalid image boundary expected outcome PASS "<<mode<<'\n';return 0;
}catch(const std::exception& e){record["probe_failure"]=e.what();if(!output.empty())std::ofstream(output)<<record.dump(2)<<'\n';std::cerr<<e.what()<<'\n';return 1;}}
