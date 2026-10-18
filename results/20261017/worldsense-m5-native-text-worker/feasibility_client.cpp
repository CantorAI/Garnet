#include "xlang3/xlang3.h"
#include <chrono>
#include <string>
#include <iostream>
#include <stdexcept>
static X::Value invoke(const X::Value& fn,const std::vector<X::Value>& args={}){X::Value out;if(!fn.Call(args,out)){const char* error=x3_runtime_last_error(fn.runtime());throw std::runtime_error(error&&*error?error:"native call failed");}return out;}
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void require(bool x,const char* reason){if(!x)throw std::runtime_error(reason);}
int main(int argc,char**argv){try{
 require(argc==4,"expected endpoint,pid,birthnonce");X::Runtime runtime;auto service=runtime.ImportRemote("garnet_text_probe",argv[1]);
 require(invoke(service["pid"]).ToLongLong()==std::stoll(argv[2]),"wrong native worker PID");require(invoke(service["identity"]).ToString()==argv[3],"wrong birth identity");
 std::string payload(70000,'x');payload+=u8"\nquote\" unicode猫🐘";require(invoke(service["echo"],{X::Value::String(runtime.host(),payload)}).ToString()==payload,"UTF8/large/newline payload changed");
 for(int i=0;i<10;++i){double before=now(),child=invoke(service["clock"]).ToDouble(),after=now();require(child>=before-.0001&&child<=after+.0001,"native/XLang monotonic epochs disagree");}
 bool propagated=false;try{invoke(service["fail"]);}catch(const std::exception& e){propagated=std::string(e.what()).find("ws58_expected_error")!=std::string::npos;}
 require(propagated,"remote native exception lost");require(invoke(service["echo"],{X::Value::String(runtime.host(),"recovered")}).ToString()=="recovered","error poisoned connection");
 require(invoke(service["stop"]).ToLongLong()!=0,"shutdown rejected");std::cout<<"NATIVE_CPP_XLANG3_LRPC_OWNER_UTF8_CLOCK_ERROR_RECOVERY_SHUTDOWN_PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
