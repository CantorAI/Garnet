from pathlib import Path
import json,hashlib,shutil,ast
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=root/'work/worldsense-native-text-worker';s=r/'out/research/worldsense-native-text-worker/source';seed=r/'out/research/worldsense-isolated-primary/source';m=json.loads((root/'work/worldsense-isolated-primary/correctness-source-manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert not s.exists()and len(m)==1718 and all(sha(seed/k)==v for k,v in m.items());s.mkdir(parents=True)
for rel in m:(s/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/rel,s/rel)
(w/'ws54-seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(s/'src/entry/native_worker_support.h').write_text('''#pragma once
#include <string>
#include <cstdint>
namespace Garnet::NativeWorkerSupport {
std::string Info(const std::string& sourceRoot);
bool WatchParent(int64_t parent);
}
''')
(s/'src/entry/native_worker_support.cpp').write_text('''// Native process identity and ownership only; no model or Metal math.
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
''')
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();needle='add_library(garnet SHARED src/entry/metal_main.cpp';assert t.count(needle)==1;p.write_text(t.replace(needle,needle+' src/entry/native_worker_support.cpp',1))
p=s/'src/entry/metal_main.cpp';t=p.read_text().replace('#include "xlang3/xlang3.h"','#include "xlang3/xlang3.h"\n#include "native_worker_support.h"',1);needle='    std::string PrimaryStatsJson() {';assert t.count(needle)==1;t=t.replace(needle,'''    std::string NativeWorkerSupportJson(){return NativeWorkerSupport::Info(GARNET_METAL_SOURCE_ROOT);}
    bool NativeWorkerWatchParent(X::Value parent){return NativeWorkerSupport::WatchParent(parent.ToLongLong());}
'''+needle,1);needle='        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);';assert t.count(needle)==1;t=t.replace(needle,needle+'''
        APISET().AddFunc<0>("native_worker_support_json",&GarnetMetalAPI::NativeWorkerSupportJson);
        APISET().AddFunc<1>("native_worker_watch_parent",&GarnetMetalAPI::NativeWorkerWatchParent);''',1);p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text();assert t.count('_models = {}')==1;t=t.replace('_models = {}',"_native_gateway_clock = 'native_text_worker' in garnet.backend_info()\n_models = {}",1)
a=t.index('        return dict(input_prompt_sha256=');b=t.index('\n\n\ndef serve_model',a);original=t[a:b];assert original.count('return dict(')==1;replacement=original.replace('        return dict(','        result = dict(',1)+'''\n        if _native_gateway_clock:
            result['native_model_start_monotonic'] = start
            result['native_first_token_monotonic'] = first
        return result''';t=t[:a]+replacement+t[b:];ast.parse(t);p.write_text(t)
# Narrow inverse proves no inference expression beyond the opt-in metadata changed.
assert t.replace("_native_gateway_clock = 'native_text_worker' in garnet.backend_info()\n",'',1).replace(replacement,original,1)==(seed/'runtime/metal_serving.py').read_text()
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};changes=[k for k,v in manifest.items()if m.get(k)!=v];assert len(manifest)==1720 and len(changes)==5
(w/'support-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'support-implementation-audit.json').write_text(json.dumps(dict(phase='support_and_clock_metadata_only',source_files=1720,changed=changes,all_MSL_codecs_math_xModel_session_original_tests_CUDA_unchanged=True,original_serving_expressions_exact_after_declared_metadata_inverse=True,gateway_not_implemented=True,no_model_or_performance_result=True),indent=2)+'\n')
shutil.copytree(root/'work/worldsense-isolated-primary/build-profile',w/'build-profile');shutil.copy2(root/'work/worldsense-isolated-primary/build-profile-identity.json',w/'build-profile-identity.json')
print('WS58 support/clock source prepared; native worker gateway not yet implemented')
