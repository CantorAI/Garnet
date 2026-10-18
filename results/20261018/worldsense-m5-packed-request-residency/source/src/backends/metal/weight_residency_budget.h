#pragma once
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
namespace Garnet {
class WeightResidencyBudget {
 std::mutex mutex_;std::unordered_map<uint64_t,uint64_t> used_;uint64_t limit_;
public:
 explicit WeightResidencyBudget(uint64_t limit):limit_(limit){}
 bool Acquire(uint64_t device,uint64_t bytes){std::lock_guard<std::mutex> lock(mutex_);if(!bytes||bytes>limit_)return false;auto& n=used_[device];if(n>limit_-bytes)return false;n+=bytes;return true;}
 void Release(uint64_t device,uint64_t bytes) noexcept {std::lock_guard<std::mutex> lock(mutex_);auto i=used_.find(device);if(i!=used_.end()&&i->second>=bytes){i->second-=bytes;if(!i->second)used_.erase(i);}}
 uint64_t Used(uint64_t device){std::lock_guard<std::mutex> lock(mutex_);auto i=used_.find(device);return i==used_.end()?0:i->second;}
};
}
