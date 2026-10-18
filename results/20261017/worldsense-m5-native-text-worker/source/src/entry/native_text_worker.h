// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
namespace Garnet {
class NativeTextWorker {
 struct Impl;std::unique_ptr<Impl> impl_;
public:
 static std::recursive_mutex& Gate();
 static double Now();
 explicit NativeTextWorker(X3Runtime* runtime);
 ~NativeTextWorker();
 bool UsesRuntime(X3Runtime* runtime)const;
 NativeTextWorker(const NativeTextWorker&)=delete;
 NativeTextWorker& operator=(const NativeTextWorker&)=delete;
 std::string Serve(X::Value root,X::Value xmodel,X::Value cache,X::Value profile,X::Value model,const std::function<std::string()>& local);
 std::string Status(X::Value model,const std::function<std::string()>& local);
 std::string Policy(X::Value enabled,const std::function<std::string()>& local);
 std::string Infer(X::Value prompt,X::Value image,X::Value limit,X::Value model,double start,const std::function<std::string()>& local,const std::function<std::string()>& parentMemory,const std::function<std::string()>& parentPrimary,const std::string& parentBackend);
 std::string State(X::Value model);
 std::string Close(X::Value model);
 std::string Info();
 void Shutdown();
};
}
