// SPDX-License-Identifier: Apache-2.0
#include "gated_delta.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
namespace Garnet {
bool GatedDeltaStep(const float* query, const float* key, const float* value,
    const float* logDecay, const float* beta, const unsigned char* active,
    std::size_t batch, std::size_t heads, std::size_t keyDim,
    std::size_t valueDim, float* state, float* output, std::string& error) {
    if (!query || !key || !value || !logDecay || !beta || !state || !output ||
        !batch || !heads || !keyDim || !valueDim) {
        error="gated delta requires nonempty dimensions and buffers"; return false;
    }
    std::size_t count=1;
    for (auto dim : {batch,heads,keyDim,valueDim}) {
        if (dim>std::numeric_limits<std::size_t>::max()/count) {
            error="gated delta dimensions overflow"; return false;
        }
        count*=dim;
    }
    std::vector<float> normalizedKey(keyDim), normalizedQuery(keyDim), delta(valueDim);
    const float queryScale=1.0f/std::sqrt(static_cast<float>(keyDim));
    for (std::size_t b=0;b<batch;++b) {
        if (active && !active[b]) {
            std::fill(output+b*heads*valueDim,output+(b+1)*heads*valueDim,0.0f);
            continue;
        }
        for (std::size_t h=0;h<heads;++h) {
            const auto bh=b*heads+h;
            const auto q=query+bh*keyDim; const auto k=key+bh*keyDim;
            auto s=state+bh*keyDim*valueDim; auto out=output+bh*valueDim;
            float q2=0,k2=0;
            for(std::size_t i=0;i<keyDim;++i){q2+=q[i]*q[i];k2+=k[i]*k[i];}
            const float qi=queryScale/std::sqrt(q2+1e-6f);
            const float ki=1.0f/std::sqrt(k2+1e-6f);
            for(std::size_t i=0;i<keyDim;++i){normalizedQuery[i]=q[i]*qi;normalizedKey[i]=k[i]*ki;}
            const float decay=std::exp(logDecay[bh]);
            for(std::size_t i=0;i<keyDim*valueDim;++i)s[i]*=decay;
            for(std::size_t v=0;v<valueDim;++v){
                float prediction=0;
                for(std::size_t i=0;i<keyDim;++i)prediction+=normalizedKey[i]*s[i*valueDim+v];
                delta[v]=(value[bh*valueDim+v]-prediction)*beta[bh];
            }
            std::fill(out,out+valueDim,0.0f);
            for(std::size_t i=0;i<keyDim;++i)for(std::size_t v=0;v<valueDim;++v){
                s[i*valueDim+v]+=normalizedKey[i]*delta[v];
                out[v]+=normalizedQuery[i]*s[i*valueDim+v];
            }
        }
    }
    error.clear(); return true;
}
}
