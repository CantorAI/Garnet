#pragma once
#include <string>
#include <cstdint>
#include <memory>
namespace Garnet::NativeWorkerSupport {
std::string Info(const std::string& sourceRoot);
bool WatchParent(int64_t parent);
std::string ThreadQoS();
class QoSScope {
 struct Impl;std::unique_ptr<Impl> impl_;
public:
 QoSScope();~QoSScope();
 QoSScope(const QoSScope&)=delete;QoSScope& operator=(const QoSScope&)=delete;
 bool Close();std::string Info()const;
};
}
