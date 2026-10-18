#include "xlang3/xlang3.h"
#include "nlohmann/json.hpp"
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/qos.h>
#include <unistd.h>
#include <cerrno>
#include <stdexcept>
#include <string>
using J=nlohmann::json;
namespace {
thread_local int active=0,starts=0,ends=0,endErrors=0;
J report(){qos_class_t q=QOS_CLASS_UNSPECIFIED;int relative=0;const int status=pthread_get_qos_class_np(pthread_self(),&q,&relative);uint64_t id=0;const int tid=pthread_threadid_np(nullptr,&id);return J{{"pid",getpid()},{"parent_pid",getppid()},{"thread_id",id},{"thread_id_status",tid},{"requested_class",static_cast<int>(q)},{"class_self",static_cast<int>(qos_class_self())},{"class_main",static_cast<int>(qos_class_main())},{"relative_priority",relative},{"get_status",status},{"active_owned_scopes",active},{"starts",starts},{"ends",ends},{"end_errors",endErrors}};}
struct Lease {
 pthread_override_t token=nullptr;
 Lease(){token=pthread_override_qos_class_start_np(pthread_self(),QOS_CLASS_USER_INITIATED,0);if(!token)throw std::runtime_error("native QoS start failed errno="+std::to_string(errno));++active;++starts;}
 void Close()noexcept{if(!token)return;auto t=token;token=nullptr;const int status=pthread_override_qos_class_end_np(t);--active;++ends;if(status)++endErrors;}
 ~Lease(){Close();}
 Lease(const Lease&)=delete;Lease& operator=(const Lease&)=delete;
};
class API {
public:
 std::string Report(){return report().dump();}
 std::string Probe(){J result;result["before"]=report();if(active!=0)throw std::runtime_error("Unexpected live scope before diagnostic");
 {Lease outer;result["during"]=report();{Lease inner;result["nested"]=report();}result["after_nested"]=report();outer.Close();outer.Close();result["after_idempotent_close"]=report();}
 bool caught=false;try{Lease lease;throw std::runtime_error("expected_qos_scope_unwind");}catch(const std::exception&e){if(std::string(e.what())!="expected_qos_scope_unwind")throw;caught=true;}
 result["after_exception"]=report();result["expected_exception_caught"]=caught;if(!caught||active!=0||endErrors!=0||starts!=ends)throw std::runtime_error("Native QoS scope cleanup failed");if(result["before"]["requested_class"]!=result["during"]["requested_class"]||result["before"]["requested_class"]!=result["after_exception"]["requested_class"])throw std::runtime_error("Override unexpectedly changed requested class");result["native_API_scope_contract_pass"]=true;result["target_class"]=static_cast<int>(QOS_CLASS_USER_INITIATED);result["scope"]="API acceptance/ownership/cleanup only; requested getters do not expose overrides. No model/GPU/performance inference.";return result.dump();}
 BEGIN_PACKAGE(API)
  APISET().AddFunc<0>("report",&API::Report);
  APISET().AddFunc<0>("probe",&API::Probe);
 END_PACKAGE
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version=X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* rawHost,X3Value current){auto* host=static_cast<X3PackageHost*>(rawHost);if(!host||host->abi_version!=X3_ABI_VERSION)return X3_STATUS_ERROR;try{API::BuildAPI();return API::APISET().Create(host,"qos_probe",current);}catch(...){return X3_STATUS_ERROR;}}
