// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "xlang3/xlang3.h"
#include "../../cuda/qwen35_packed_commit.cuh"
#include <mutex>
#include <string>
namespace Garnet {
class CUDAPackedPrefix {
    Qwen35PackedCommitConfig config_{};
    X::Tensor linear_,full_,parameters_,workspace_;
    bool ready_=false;
    mutable std::mutex mutex_;
    void CloseUnlocked();
public:
    ~CUDAPackedPrefix();
    bool Build(const std::string& checkpoint,const std::string& specification);
    void Commit(X::Value packed,X::Value canonical,X::Value spans);
    std::string StatsJson() const;
    void Close();
    BEGIN_PACKAGE(CUDAPackedPrefix)
        APISET().AddFunc<2>("build",&CUDAPackedPrefix::Build);
        APISET().AddFunc<3>("commit",&CUDAPackedPrefix::Commit);
        APISET().AddFunc<0>("stats_json",&CUDAPackedPrefix::StatsJson);
        APISET().AddFunc<0>("close",&CUDAPackedPrefix::Close);
    END_PACKAGE
};
}
#endif
