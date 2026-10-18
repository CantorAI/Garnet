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
}
