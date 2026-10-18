#include "native_worker_support.h"
#include "nlohmann/json.hpp"
#include <thread>
#include <iostream>
#include <stdexcept>
using namespace Garnet::NativeWorkerSupport;using J=nlohmann::json;
void one(){auto before=J::parse(ThreadQoS());if(before["active_scopes"]!=0)throw std::runtime_error("Unexpected scope");{QoSScope a;{QoSScope b;if(J::parse(ThreadQoS())["active_scopes"]!=2)throw std::runtime_error("Nested count");}if(!a.Close()||!a.Close())throw std::runtime_error("Close");}{QoSScope fallback;}
 bool caught=false;try{QoSScope unwind;throw std::runtime_error("expected_unwind");}catch(const std::exception&e){caught=std::string(e.what())=="expected_unwind";}
 auto after=J::parse(ThreadQoS());if(!caught||after["active_scopes"]!=0||after["end_errors"]!=0||after["starts"]!=after["ends"])throw std::runtime_error("Fallback/unwind leaked scope");if(before["requested_class"]!=after["requested_class"])throw std::runtime_error("Requested class changed");}
int main(){try{one();std::exception_ptr error;std::thread worker([&]{try{one();}catch(...){error=std::current_exception();}});worker.join();if(error)std::rethrow_exception(error);std::cout<<"Native QoS normal/nested/idempotent/destructor/unwind across main/stdthread PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
