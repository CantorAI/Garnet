#pragma once
#include <string>
#include <cstdint>
namespace Garnet::NativeWorkerSupport {
std::string Info(const std::string& sourceRoot);
bool WatchParent(int64_t parent);
}
