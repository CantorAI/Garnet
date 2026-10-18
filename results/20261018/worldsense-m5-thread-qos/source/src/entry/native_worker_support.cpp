// Native process identity and ownership only; no model or Metal math.
#include "native_worker_support.h"
#include "nlohmann/json.hpp"
#include <unistd.h>
#include <dlfcn.h>
#include <cstdlib>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/qos.h>
#include <cerrno>
namespace Garnet::NativeWorkerSupport {
namespace {
struct Watchdog {
 std::mutex mutex;std::atomic<bool> stop{false};std::thread thread;pid_t parent=0;
 ~Watchdog(){stop.store(true);if(thread.joinable())thread.join();}
};
Watchdog& State(){static Watchdog state;return state;}
}
std::string Info(const std::string& sourceRoot){
 auto& state=State();std::lock_guard<std::mutex> lock(state.mutex);Dl_info image{};
 if(!dladdr(reinterpret_cast<const void*>(&Info),&image)||!image.dli_fname)throw std::runtime_error("Native worker library identity unavailable");
 const double now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
 return nlohmann::json{{"pid",getpid()},{"parent_pid",getppid()},{"library_path",std::filesystem::weakly_canonical(image.dli_fname).string()},{"source_root",sourceRoot},{"steady_seconds",now},{"watch_parent_pid",state.parent},{"watchdog_active",state.thread.joinable()},{"scope","native process/library identity and common monotonic epoch; not model throughput or physical residency"}}.dump();
}
bool WatchParent(int64_t expected){
 if(expected<=1||expected!=getppid())throw std::invalid_argument("Native worker parent ownership mismatch");
 const char* marker=std::getenv("GARNET_NATIVE_WORKER_PARENT");
 if(!marker||std::string(marker)!=std::to_string(expected))throw std::invalid_argument("Native worker launch marker missing");
 auto& state=State();std::lock_guard<std::mutex> lock(state.mutex);
 if(state.thread.joinable()){if(state.parent!=expected)throw std::logic_error("Native worker parent cannot change");return true;}
 state.parent=static_cast<pid_t>(expected);state.stop.store(false);
 state.thread=std::thread([&state,parent=state.parent]{while(!state.stop.load()){if(getppid()!=parent)std::_Exit(72);std::this_thread::sleep_for(std::chrono::milliseconds(20));}});return true;
}
namespace {
std::atomic<uint64_t> qosAttempts{0},qosStarts{0},qosEnds{0},qosActive{0},qosEndErrors{0};
nlohmann::json QoSSnapshot(){
 qos_class_t requested=QOS_CLASS_UNSPECIFIED;int relative=0;uint64_t tid=0;
 const int status=pthread_get_qos_class_np(pthread_self(),&requested,&relative),threadStatus=pthread_threadid_np(nullptr,&tid);
 return {{"pid",getpid()},{"thread_id",tid},{"thread_id_status",threadStatus},{"get_status",status},{"requested_class",static_cast<int>(requested)},{"class_self",static_cast<int>(qos_class_self())},{"class_main",static_cast<int>(qos_class_main())},{"relative_priority",relative},{"start_attempts",qosAttempts.load()},{"starts",qosStarts.load()},{"ends",qosEnds.load()},{"active_scopes",qosActive.load()},{"end_errors",qosEndErrors.load()},{"scope","Requested QoS only: public getters do not expose overrides/effective GPU scheduling or frequency."}};
}
}
std::string ThreadQoS(){return QoSSnapshot().dump();}
struct QoSScope::Impl {
 pthread_override_t token=nullptr;nlohmann::json before,during,after;int endStatus=0;
 Impl(){before=QoSSnapshot();++qosAttempts;token=pthread_override_qos_class_start_np(pthread_self(),QOS_CLASS_USER_INITIATED,0);if(!token)throw std::runtime_error("Native inference QoS override start failed errno="+std::to_string(errno));++qosStarts;++qosActive;try{during=QoSSnapshot();}catch(...){Close();throw;}}
 bool Close()noexcept{if(!token)return endStatus==0;auto value=token;token=nullptr;endStatus=pthread_override_qos_class_end_np(value);++qosEnds;--qosActive;if(endStatus)++qosEndErrors;try{after=QoSSnapshot();}catch(...){}return endStatus==0;}
 ~Impl(){Close();}
};
QoSScope::QoSScope():impl_(new Impl()){}
QoSScope::~QoSScope()=default;
bool QoSScope::Close(){if(!impl_->Close())throw std::runtime_error("Native inference QoS override end failed status="+std::to_string(impl_->endStatus));if(impl_->after.is_null())throw std::runtime_error("Native inference QoS close metadata unavailable");return true;}
std::string QoSScope::Info()const{return nlohmann::json{{"target_class",static_cast<int>(QOS_CLASS_USER_INITIATED)},{"relative_priority",0},{"before",impl_->before},{"during",impl_->during},{"after",impl_->after},{"closed",impl_->token==nullptr},{"api_start_accepted",true},{"api_end_status",impl_->endStatus},{"scope","Owned native current-thread QoS dependency scope; requested getters do not report the override's effective scheduling."}}.dump();}

}
