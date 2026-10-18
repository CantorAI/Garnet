from pathlib import Path
import json,hashlib,shutil,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-native-text-worker/source';s=r/'out/research/worldsense-thread-qos/source';m=json.loads((w.parent/'worldsense-native-text-worker/gateway-source-manifest.json').read_text());assert len(m)==1723 and all(hashlib.sha256((old/k).read_bytes()).hexdigest()==v for k,v in m.items());assert not s.exists();s.mkdir(parents=True)
for k in m:p=s/k;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/k,p)
(w/'seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
# Native public API scope; no math/Metal/SDK dependencies.
p=s/'src/entry/native_worker_support.h';t=p.read_text().replace('#include <cstdint>','#include <cstdint>\n#include <memory>');t=t.replace('bool WatchParent(int64_t parent);','''bool WatchParent(int64_t parent);
std::string ThreadQoS();
class QoSScope {
 struct Impl;std::unique_ptr<Impl> impl_;
public:
 QoSScope();~QoSScope();
 QoSScope(const QoSScope&)=delete;QoSScope& operator=(const QoSScope&)=delete;
 bool Close();std::string Info()const;
};''');p.write_text(t)
p=s/'src/entry/native_worker_support.cpp';t=p.read_text().replace('#include <stdexcept>','#include <stdexcept>\n#include <pthread.h>\n#include <pthread/qos.h>\n#include <sys/qos.h>\n#include <cerrno>');at=t.rfind('\n}');t=t[:at]+'''
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
 Impl(){before=QoSSnapshot();++qosAttempts;token=pthread_override_qos_class_start_np(pthread_self(),QOS_CLASS_USER_INITIATED,0);if(!token)throw std::runtime_error("Native inference QoS override start failed errno="+std::to_string(errno));++qosStarts;++qosActive;during=QoSSnapshot();}
 bool Close()noexcept{if(!token)return endStatus==0;auto value=token;token=nullptr;endStatus=pthread_override_qos_class_end_np(value);++qosEnds;--qosActive;if(endStatus)++qosEndErrors;try{after=QoSSnapshot();}catch(...){}return endStatus==0;}
 ~Impl(){Close();}
};
QoSScope::QoSScope():impl_(new Impl()){}
QoSScope::~QoSScope()=default;
bool QoSScope::Close(){if(!impl_->Close())throw std::runtime_error("Native inference QoS override end failed status="+std::to_string(impl_->endStatus));return true;}
std::string QoSScope::Info()const{return nlohmann::json{{"target_class",static_cast<int>(QOS_CLASS_USER_INITIATED)},{"relative_priority",0},{"before",impl_->before},{"during",impl_->during},{"after",impl_->after},{"closed",impl_->token==nullptr},{"api_start_accepted",true},{"api_end_status",impl_->endStatus},{"scope","Owned native current-thread QoS dependency scope; requested getters do not report the override's effective scheduling."}}.dump();}
'''+t[at:];p.write_text(t)
# Local SDK object never crosses IPC, explicit close plus destructor fallback.
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='class GarnetMetalAPI {';assert t.count(needle)==1;t=t.replace(needle,'''class NativeThreadQoS {
 NativeWorkerSupport::QoSScope scope_;
public:
 bool Close(){return scope_.Close();}
 std::string Info(){return scope_.Info();}
 BEGIN_PACKAGE(NativeThreadQoS)
  APISET().AddFunc<0>("close",&NativeThreadQoS::Close);
  APISET().AddFunc<0>("info_json",&NativeThreadQoS::Info);
 END_PACKAGE
};
'''+needle,1);needle='    bool NativeWorkerWatchParent(';index=t.index(needle);t=t[:index]+'    std::string NativeThreadQoSJson(){return NativeWorkerSupport::ThreadQoS();}\n'+t[index:];needle='        APISET().AddClass<0,GarnetTensor>("tensor");';assert needle in t;t=t.replace(needle,needle+'\n        APISET().AddClass<0,NativeThreadQoS>("native_thread_qos");\n        APISET().AddFunc<0>("native_thread_qos_json",&GarnetMetalAPI::NativeThreadQoSJson);');needle='        return info;\n    }\n    // Graphs own';assert needle in t;t=t.replace(needle,'#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_QOS\n        info+="; native_text_worker_user_initiated_qos";\n#endif\n'+needle);p.write_text(t)
p=s/'src/entry/native_text_worker.cpp';t=p.read_text();needle='  std::vector<std::string> args={';index=t.index(needle);t=t[:index]+'''#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_QOS
  const std::string qosMode="1";
#else
  const std::string qosMode="0";
#endif
'''+t[index:];needle='source.string(),ready.string()};';assert t.count(needle)==1;t=t.replace(needle,'source.string(),ready.string(),qosMode};',1);needle=' const auto parentPrimaryBefore=J::parse(parentPrimary());';assert t.count(needle)==1;t=t.replace(needle,' const auto callerQoS=J::parse(NativeWorkerSupport::ThreadQoS());\n'+needle);needle=' auto parent=J::parse(parentMemory());';assert t.count(needle)==1;t=t.replace(needle,' meta["parent_thread_qos_requested"]=callerQoS;\n'+needle,1);p.write_text(t)
p=s/'runtime/native_text_worker.py';t=p.read_text();needle='sys.path.insert(0,str(source/\'runtime\'))';assert t.count(needle)==1;t=t.replace(needle,"qos_enabled=len(sys.argv)>7 and sys.argv[7]=='1'\n"+needle,1);t=t.replace('def __init__(self):self.running=True;self.infer_calls=0','def __init__(self):self.running=True;self.infer_calls=0;self.last_inference_qos=None');needle='infer_calls=self.infer_calls))';assert t.count(needle)==1;t=t.replace(needle,"infer_calls=self.infer_calls,inference_qos_mode='user_initiated'if qos_enabled else'inherited',last_inference_qos=self.last_inference_qos))");old="""  self.infer_calls+=1
  result=json.loads(garnet.infer_json(prompt,image,limit,name));result['native_worker_call_index']=self.infer_calls
  return json.dumps(result)""";assert t.count(old)==1;new="""  self.infer_calls+=1
  before=json.loads(garnet.native_thread_qos_json());lease=None;scope=None
  try:
   if qos_enabled:lease=garnet.native_thread_qos();scope=json.loads(lease.info_json())
   result=json.loads(garnet.infer_json(prompt,image,limit,name))
  finally:
   if lease is not None:
    assert lease.close();scope=json.loads(lease.info_json())
   after=json.loads(garnet.native_thread_qos_json())
   self.last_inference_qos=dict(mode='user_initiated'if qos_enabled else'inherited',before=before,scope=scope,after=after)
   if after['active_scopes']!=0 or after['end_errors']!=0:raise ValueError('Native inference QoS scope did not close cleanly')
  result['native_worker_call_index']=self.infer_calls
  result['native_worker_qos']=self.last_inference_qos
  return json.dumps(result)""";t=t.replace(old,new);p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();t+='''
option(GARNET_METAL_NATIVE_TEXT_WORKER_QOS "Bound native text inference by a user-initiated thread QoS dependency scope" OFF)
if(GARNET_METAL_NATIVE_TEXT_WORKER_QOS)
 if(NOT GARNET_METAL_NATIVE_TEXT_WORKER)
  message(FATAL_ERROR "Native text QoS requires the owned native text worker")
 endif()
 target_compile_definitions(garnet PRIVATE GARNET_METAL_NATIVE_TEXT_WORKER_QOS=1)
endif()
''';p.write_text(t)
manifest={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()};changed=sorted(k for k,v in manifest.items()if m.get(k)!=v);assert len(manifest)==1723 and changed==sorted(['src/entry/native_worker_support.h','src/entry/native_worker_support.cpp','src/entry/metal_main.cpp','src/entry/native_text_worker.cpp','runtime/native_text_worker.py','cmake/GarnetMetal.cmake']);(w/'initial-implementation-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'initial-implementation-audit.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed_files=1723,source_files=1723,changed_files=changed,all_other_source_exact58=True,no_shader_math_model_session_CUDA_engine_SDK_change=True,no_build_or_model_run_yet=True),indent=2)+'\n');print('New isolated WS59 source1723/sixfile delta implemented; original shader/math/model/session/CUDA source exact58')
