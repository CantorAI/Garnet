// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
#include "gpt_oss_flash_prefill.h"
#endif
#include "gpt_oss_marlin.h"
#include "gpt_oss_router_dispatch.h"
#include "gpt_oss_weight_shard.h"
#include "gpt_oss_marlin_pack.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>
using namespace Garnet;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
float bf(float x) { uint32_t v; std::memcpy(&v, &x, 4); v += 0x7fff + ((v >> 16) & 1); v &= 0xffff0000; std::memcpy(&x, &v, 4); return x; }
uint16_t bits(float x) { x = bf(x); uint32_t v; std::memcpy(&v, &x, 4); return uint16_t(v >> 16); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(const std::vector<T>& v) : n(v.size()) {
        check(cudaMalloc((void**)&p, n * sizeof(T))); check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(p); }
    std::vector<T> read() { std::vector<T> v(n); check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
template<class T> std::vector<T> shardExpertRows(const std::vector<T>& original, int experts, int rank) {
    std::vector<unsigned char> bytes;
    std::string error;
    if (!GatherGptOssExpertShard(original.data(), original.size() * sizeof(T),
            experts, rank, bytes, error)) throw std::runtime_error(error);
    std::vector<T> result(bytes.size() / sizeof(T));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}
template<class T> std::vector<T> shardIntermediate(const std::vector<T>& original,
    const std::vector<long long>& shape,int rank,int mode) {
    std::vector<unsigned char> bytes;std::string error;
    if(!GatherGptOssIntermediateShard(original.data(),original.size()*sizeof(T),shape,
            sizeof(T),rank,mode,bytes,error))throw std::runtime_error(error);
    std::vector<T> result(bytes.size()/sizeof(T));std::memcpy(result.data(),bytes.data(),bytes.size());return result;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const char* name) {
    if (actual.size() != expected.size()) throw std::runtime_error("size mismatch");
    float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float error = std::abs(actual[i] - expected[i]); maximum = std::max(maximum, error);
        if (!std::isfinite(actual[i]) || error > tolerance * (1 + std::abs(expected[i]))) {
            std::cerr << name << " mismatch " << i << ": " << actual[i] << " expected " << expected[i] << "\n";
            throw std::runtime_error(name);
        }
    }
    std::cout << name << " passed; max absolute error " << maximum << "\n";
}
void testDecodeGemv(int outputs, int inputs) {
    std::vector<uint16_t> x(inputs), weight(size_t(outputs) * inputs);
    for (int col = 0; col < inputs; ++col)
        x[col] = bits(std::sin(float(col) * .013f) * .1f);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = bits(std::cos(float(i) * .017f) * .1f);
    Device<uint16_t> dx(x), dw(weight), dy{std::vector<uint16_t>(outputs)};
    check(RunGptOssDecodeGemv(dx.p, dw.p, dy.p, outputs, inputs,
        inputs, 0, 0, 0, 0, -1, nullptr));
    std::vector<float> expected(outputs), actual(outputs);
    const auto result = dy.read();
    for (int row = 0; row < outputs; ++row) {
        float sum = 0;
        for (int col = 0; col < inputs; ++col) {
            uint32_t xb = uint32_t(x[col]) << 16, wb =
                uint32_t(weight[size_t(row) * inputs + col]) << 16;
            float xf, wf;
            std::memcpy(&xf, &xb, sizeof(xf));
            std::memcpy(&wf, &wb, sizeof(wf));
            sum = std::fma(xf, wf, sum);
        }
        expected[row] = bf(sum);
        uint32_t yb = uint32_t(result[row]) << 16;
        std::memcpy(&actual[row], &yb, sizeof(float));
    }
    compare(actual, expected, .008f, "GPT-OSS BF16 decode GEMV");
}
void testDecodeGemvSharded(bool qkv, int rank) {
    const int inputs = qkv ? 2880 : 2048;
    const int outputs = qkv ? 2560 : 2880;
    const int fullRows = qkv ? 5120 : 2880;
    const int fullColumns = qkv ? 2880 : 4096;
    std::vector<uint16_t> x(inputs), weight(size_t(fullRows) * fullColumns);
    for (int col = 0; col < inputs; ++col)
        x[col] = bits(std::sin(float(col) * .013f) * .1f);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = bits(std::cos(float(i) * .017f) * .1f);
    Device<uint16_t> dx(x), dw(weight), dy{std::vector<uint16_t>(outputs)};
    check(RunGptOssDecodeGemv(dx.p, dw.p, dy.p, outputs, inputs,
        fullColumns, qkv ? 0 : rank * inputs,
        qkv ? 64 : 0, qkv ? 8 : 0, qkv ? 64 : 0, rank, nullptr));
    const auto result = dy.read();
    const std::vector<int> sampled = qkv
        ? std::vector<int>{0, 2047, 2048, 2303, 2304, 2559}
        : std::vector<int>{0, 1, 1439, 2879};
    std::vector<float> expected, actual;
    for (int row : sampled) {
        int weightRow = row;
        if (qkv) {
            if (row < 2048) weightRow = rank * 2048 + row;
            else if (row < 2304) weightRow = 4096 + rank * 256 + row - 2048;
            else weightRow = 4608 + rank * 256 + row - 2304;
        }
        const int columnOffset = qkv ? 0 : rank * inputs;
        float sum = 0;
        for (int col = 0; col < inputs; ++col) {
            uint32_t xb = uint32_t(x[col]) << 16;
            uint32_t wb = uint32_t(weight[size_t(weightRow) * fullColumns +
                columnOffset + col]) << 16;
            float xf, wf;
            std::memcpy(&xf, &xb, sizeof(xf));
            std::memcpy(&wf, &wb, sizeof(wf));
            sum = std::fma(xf, wf, sum);
        }
        expected.push_back(bf(sum));
        uint32_t yb = uint32_t(result[row]) << 16;
        float yf;
        std::memcpy(&yf, &yb, sizeof(yf));
        actual.push_back(yf);
    }
    compare(actual, expected, .008f,
        qkv ? "GPT-OSS TP2 QKV GEMV" : "GPT-OSS TP2 row GEMV");
}
void testRmsNorm() {
    constexpr int tokens = 19, hidden = 2880;
    constexpr float epsilon = 1.0e-5f;
    std::vector<float> x(size_t(tokens) * hidden), weight(hidden), expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = bf(std::sin(float(i) * .017f) * 1.7f + std::cos(float(i) * .003f));
    for (int d = 0; d < hidden; ++d)
        weight[d] = bf(.75f + .5f * std::sin(float(d) * .011f));
    for (int row = 0; row < tokens; ++row) {
        float squareSum = 0.f;
        for (int d = 0; d < hidden; ++d) {
            const float value = bf(x[size_t(row) * hidden + d]);
            squareSum += value * value;
        }
        const float inverse = 1.f / std::sqrt(squareSum / hidden + epsilon);
        for (int d = 0; d < hidden; ++d) {
            const float value = bf(x[size_t(row) * hidden + d]);
            expected[size_t(row) * hidden + d] = bf((value * inverse) * weight[d]);
        }
    }
    Device<float> dx(x), dw(weight), dy(std::vector<float>(x.size()));
    for (int threads : {128, 256, 512, 1024}) {
        check(RunGptOssRmsNorm(dx.p, dw.p, dy.p, tokens, hidden, epsilon, threads, nullptr));
        // The CUDA tree reduction and the CPU reference sum in different orders;
        // permit at most one BF16-scale rounding step while retaining a tight bound.
        compare(dy.read(), expected, .008f, "GPT-OSS BF16 RMSNorm");
    }
}
void testRope() {
    GptOssOptions o; o.qHeads = 4; o.kvHeads = 2; o.headDim = 64;
    const int width = 512, tokens = 4;
    std::vector<float> x(tokens * width), expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf(std::sin(float(i) * .1f));
    std::vector<int64_t> positions{0, 127, 4096, 90000};
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < width; ++c) {
        size_t i = size_t(t) * width + c;
        if (c >= 384) { expected[i] = x[i]; continue; }
        int d = c % 64, f = d % 32;
        double low = 32 * std::log(4096. / (32 * 2 * 3.141592653589793)) / std::log(150000.);
        double high = 32 * std::log(4096. / (2 * 3.141592653589793)) / std::log(150000.);
        double ramp = std::clamp((f - low) / (high - low), 0., 1.);
        double inv = std::pow(150000., -2. * f / 64) * ((1 - ramp) + ramp / 32);
        double angle = positions[t] * inv, concentration = 1 + .1 * std::log(32.);
        float rotated = d < 32 ? -x[i + 32] : x[i - 32];
        expected[i] = bf(bf(x[i] * bf(float(std::cos(angle) * concentration))) + bf(rotated * bf(float(std::sin(angle) * concentration))));
    }
    Device<float> dx(x), dy(std::vector<float>(x.size())); Device<int64_t> dp(positions);
    check(RunGptOssRope(dx.p, dp.p, dy.p, tokens, o, nullptr));
    compare(dy.read(), expected, .012f, "YaRN (including long positions and unchanged V)");
}
void testAttention(int dimension) {
    GptOssOptions o; o.kind = 1; o.qHeads = 4; o.kvHeads = 2; o.headDim = dimension; o.pageSize = 2; o.layer = 1; o.prefill = 1;
    const int batch = 2, tokens = 3, pages = 4, logical = 2, width = 8 * dimension;
    const int queryWidth = 4 * dimension, valueStart = 6 * dimension;
    std::vector<float> x(batch * tokens * width), sinks{.4f, -.2f, 2, -1};
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf(std::sin(float(i) * .13f));
    std::vector<uint16_t> keys(2 * pages * 2 * 2 * dimension, bits(17)), values(keys);
    std::vector<int> table{2, 0, 3, 1}, lengths{3, 3}, starts{0, 0}, active{1, 1};
    Device<float> dx(x), ds(sinks), dy(std::vector<float>(batch * tokens * queryWidth));
    Device<float> scratch(std::vector<float>(size_t(batch) * o.qHeads * 64 * 130));
    Device<uint16_t> dk(keys), dv(values);
    Device<int> dt(table), dl(lengths), dp(starts), da(active);
    const void* in[]{dx.p, dk.p, dv.p, dt.p, dl.p, dp.p, da.p, ds.p};
    for (int window : {0, 2}) {
        o.window = window;
        check(RunGptOssAttention(in, dy.p, scratch.p, batch, tokens, logical, pages, o, nullptr));
        std::vector<float> expected(batch * tokens * queryWidth);
        for (int b = 0; b < batch; ++b) for (int t = 0; t < tokens; ++t) for (int h = 0; h < 4; ++h) {
            int first = window ? std::max(0, t + 1 - window) : 0;
            double sum = std::exp(double(sinks[h])); std::vector<double> accum(dimension);
            for (int p = first; p <= t; ++p) {
                double score = 0;
                for (int d = 0; d < dimension; ++d) score += double(x[(b * tokens + t) * width + h * dimension + d]) * x[(b * tokens + p) * width + queryWidth + (h / 2) * dimension + d];
                double weight = std::exp(score / std::sqrt(double(dimension))); sum += weight;
                for (int d = 0; d < dimension; ++d) accum[d] += weight * x[(b * tokens + p) * width + valueStart + (h / 2) * dimension + d];
            }
            for (int d = 0; d < dimension; ++d) expected[(b * tokens + t) * queryWidth + h * dimension + d] = bf(float(accum[d] / sum));
        }
        compare(dy.read(), expected, .006f, window ? "Sliding attention + sinks + batch isolation" : "Full attention + sinks + batch isolation");
    }
    // Cached single-token decode must match the last prefill token, with inactive
    // slots producing zero attention and leaving their cache pages untouched.
    auto savedK = dk.read(), savedV = dv.read();
    std::vector<float> decodeInput(batch * width);
    for (int b = 0; b < batch; ++b) std::copy_n(x.begin() + (b * tokens + 2) * width, width, decodeInput.begin() + b * width);
    Device<float> dd(decodeInput), dout(std::vector<float>(batch * queryWidth));
    Device<int> decodeStarts(std::vector<int>{2, 2}), masked(std::vector<int>{1, 0});
    const void* decode[]{dd.p, dk.p, dv.p, dt.p, dl.p, decodeStarts.p, masked.p, ds.p};
    auto prefill = dy.read(); o.prefill = 0;
    check(RunGptOssAttention(decode, dout.p, scratch.p, batch, 1, logical, pages, o, nullptr));
    auto actual = dout.read(); std::vector<float> expected(batch * queryWidth, 0);
    std::copy_n(prefill.begin() + 2 * queryWidth, queryWidth, expected.begin());
    compare(actual, expected, .006f, "Cached decode + inactive slot");
    if (dk.read() != savedK || dv.read() != savedV) throw std::runtime_error("decode cache mutation mismatch");
    for (size_t i = 0; i < keys.size() / 2; ++i) if (savedK[i] != keys[i]) throw std::runtime_error("wrong KV layer modified");
}
void testHybridRingAttention64() {
    // Independent full-history physical storage plus FP64 sampled queries.
    // Never form the oracle from the ring's potentially overwritten entries.
    for(int window:{17,128})for(int tokens:{1,16,32})for(int start:{0,127,2005})
      for(int heads:{8,32})for(int layer:{0,1}) {
        std::cout<<"Hybrid case window="<<window<<" tokens="<<tokens<<" start="<<start
                 <<" heads="<<heads<<" layer="<<layer<<std::endl;
        GptOssOptions o;o.kind=1;o.qHeads=heads;o.kvHeads=heads/8;
        o.headDim=64;o.pageSize=16;o.layer=layer;o.prefill=tokens>1;o.window=window;
        constexpr int batch=3,logical=160,layers=2;
        const int ring=(window+tokens-1+15)/16,fullPages=batch*logical,ringPages=batch*ring;
        const int width=(heads+2*o.kvHeads)*64,queryWidth=heads*64;
        const std::vector<int> starts{start,start+3,start},active{1,1,0};
        const std::vector<int> lengths{start+tokens,start+3+std::max(1,tokens-3),0};
        std::vector<int> fullTable(batch*logical),ringTable(batch*logical);
        for(int b=0;b<batch;++b)for(int p=0;p<logical;++p) {
            fullTable[b*logical+p]=b*logical+p;ringTable[b*logical+p]=b*ring+p%ring;
        }
        std::vector<uint16_t> fullK(size_t(layers)*fullPages*16*o.kvHeads*64,bits(7.f)),fullV(fullK);
        std::vector<uint16_t> ringK(size_t(layers)*ringPages*16*o.kvHeads*64,bits(7.f)),ringV(ringK);
        auto offset=[&](int l,int pages,int page,int p,int h,int d) {
            return (((size_t(l)*pages+page)*16+p%16)*o.kvHeads+h)*64+d;
        };
        for(int l=0;l<layers;++l)for(int b=0;b<batch;++b)for(int p=0;p<starts[b];++p)
          for(int h=0;h<o.kvHeads;++h)for(int d=0;d<64;++d) {
            const float index=float(31*l+17*b+11*p+5*h+d);
            const uint16_t k=bits(std::sin(index*.031f)*.75f),v=bits(std::cos(index*.021f)*.75f);
            const auto a=offset(l,fullPages,b*logical+p/16,p,h,d);
            const auto c=offset(l,ringPages,b*ring+(p/16)%ring,p,h,d);
            fullK[a]=ringK[c]=k;fullV[a]=ringV[c]=v;
        }
        std::vector<float> x(size_t(batch)*tokens*width),sinks(heads);
        for(int h=0;h<heads;++h)sinks[h]=(h%4==0)?12.f:(h%4==1)?-12.f:(h%4==2)?2.f:-1.f;
        Device<float> dx(x),ds(sinks),fullOut{std::vector<float>(size_t(batch)*tokens*queryWidth)},
            ringOut{std::vector<float>(fullOut.n)},scratch{std::vector<float>(size_t(batch)*heads*64*130)};
        Device<uint16_t> dfk(fullK),dfv(fullV),drk(ringK),drv(ringV);
        Device<int> dft(fullTable),drt(ringTable),dl(lengths),dp(starts),da(active);
        const void* fullIn[]{dx.p,dfk.p,dfv.p,dft.p,dl.p,dp.p,da.p,ds.p};
        const void* ringIn[]{dx.p,drk.p,drv.p,drt.p,dl.p,dp.p,da.p,ds.p};
        // Device constructors upload on the default stream. Complete those
        // initial transfers before a nonblocking execution stream can read.
        check(cudaStreamSynchronize(nullptr));
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        // Gate script disables automatic FlashInfer selection for the exact
        // storage comparison. Its independent explicit path is checked below.
        check(RunGptOssAttention(ringIn,ringOut.p,scratch.p,batch,tokens,logical,ringPages,o,stream));
        check(cudaStreamSynchronize(stream));
        cudaGraph_t graph;cudaGraphExec_t execution;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        check(RunGptOssAttention(ringIn,ringOut.p,scratch.p,batch,tokens,logical,ringPages,o,stream));
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiate(&execution,graph,nullptr,nullptr,0));
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
        const auto bytes=GptOssFlashAttentionWorkspace(batch,tokens,logical,o);
        if(!bytes)throw std::runtime_error("Hybrid ring requires actual FlashInfer coverage");
        Device<unsigned char> flashWorkspace{std::vector<unsigned char>(bytes)};
        Device<float> flashOut{std::vector<float>(fullOut.n)};
        check(RunGptOssFlashAttention(ringIn,flashOut.p,flashWorkspace.p,batch,tokens,logical,ringPages,o,stream));
        check(cudaStreamSynchronize(stream));
        cudaGraph_t flashGraph;cudaGraphExec_t flashExecution;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        check(RunGptOssFlashAttention(ringIn,flashOut.p,flashWorkspace.p,batch,tokens,logical,ringPages,o,stream));
        check(cudaStreamEndCapture(stream,&flashGraph));
        check(cudaGraphInstantiate(&flashExecution,flashGraph,nullptr,nullptr,0));
#endif
        for(int change=0;change<2;++change) {
            for(size_t i=0;i<x.size();++i)x[i]=bf(std::sin(float(i)*.019f+change*.125f)*.75f);
            // Pageable H2D return is not a cross-stream dependency. The input
            // update must precede both full-history and ring graph execution.
            check(cudaMemcpyAsync(dx.p,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            check(RunGptOssAttention(fullIn,fullOut.p,scratch.p,batch,tokens,logical,fullPages,o,stream));
            check(cudaGraphLaunch(execution,stream));
            check(cudaStreamSynchronize(stream));
            const auto expected=fullOut.read(),actual=ringOut.read();
            const auto cachedK=dfk.read(),cachedV=dfv.read();
            const auto ringKeys=drk.read(),ringValues=drv.read();
            for(int b=0;b<batch;++b)if(active[b]) {
                const int firstEnd=o.prefill?starts[b]+1:lengths[b];
                const int lastEnd=o.prefill?starts[b]+tokens:lengths[b];
                for(int p=std::max(0,firstEnd-window);p<lastEnd;++p)
                  for(int h=0;h<o.kvHeads;++h)for(int d=0;d<64;++d) {
                    const auto a=offset(layer,fullPages,b*logical+p/16,p,h,d);
                    const auto c=offset(layer,ringPages,b*ring+(p/16)%ring,p,h,d);
                    if(cachedK[a]!=ringKeys[c]||cachedV[a]!=ringValues[c]) {
                        std::cerr<<"Hybrid visible KV mismatch change="<<change<<" slot="<<b
                                 <<" position="<<p<<" head="<<h<<" dim="<<d<<std::endl;
                        throw std::runtime_error("Hybrid ring visible KV differs from full history");
                    }
                }
            }
            compare(actual,expected,0.f,"Hybrid ring exact full-history outputs/changing graph");
            auto value=[](uint16_t v){uint32_t raw=uint32_t(v)<<16;float f;std::memcpy(&f,&raw,4);return f;};
            for(int b=0;b<batch;++b)for(int t:{0,tokens/2,tokens-1})for(int h=0;h<heads;++h) {
                const int end=o.prefill?starts[b]+t+1:lengths[b];
                double denominator=std::exp(double(sinks[h])),accum[64]={};
                if(active[b])for(int p=std::max(0,end-window);p<end;++p) {
                    double score=0;
                    for(int d=0;d<64;++d)score+=double(x[size_t(b*tokens+t)*width+h*64+d])*
                        value(cachedK[offset(layer,fullPages,b*logical+p/16,p,h/8,d)]);
                    const double weight=std::exp(score/8.);denominator+=weight;
                    for(int d=0;d<64;++d)accum[d]+=weight*value(cachedV[offset(layer,fullPages,b*logical+p/16,p,h/8,d)]);
                }
                for(int d=0;d<64;++d) {
                    const float wanted=bf(float(accum[d]/denominator));
                    if(std::abs(actual[size_t(b*tokens+t)*queryWidth+h*64+d]-wanted)>.006f*(1+std::abs(wanted)))
                        throw std::runtime_error("Hybrid ring differs from independent FP64 full history");
                }
            }
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
            check(cudaGraphLaunch(flashExecution,stream));
            check(cudaStreamSynchronize(stream));
            compare(flashOut.read(),expected,.006f,"Hybrid ring FlashInfer/changing graph vs full history");
#endif
        }
        const auto savedK=drk.read(),savedV=drv.read();
        const size_t perLayer=size_t(ringPages)*16*o.kvHeads*64;
        for(size_t i=0;i<savedK.size();++i) {
            const int l=int(i/perLayer),b=int((i%perLayer)/(size_t(ring)*16*o.kvHeads*64));
            if((l!=layer||!active[b])&&(savedK[i]!=ringK[i]||savedV[i]!=ringV[i]))
                throw std::runtime_error("Hybrid ring modified inactive request or other bank layer");
        }
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
        check(cudaGraphExecDestroy(flashExecution));check(cudaGraphDestroy(flashGraph));
#endif
        check(cudaGraphExecDestroy(execution));check(cudaGraphDestroy(graph));
        check(cudaStreamDestroy(stream));
    }
    std::cout<<"Hybrid ring repeated tables, both bank layers, FP64 sample, changing graphs and inactive storage PASS\n";
}

void testLongPrefillAttention64() {
    GptOssOptions o; o.kind = 1; o.qHeads = 4; o.kvHeads = 2;
    o.headDim = 64; o.pageSize = 16; o.layer = 1; o.prefill = 1;
    constexpr int tokens = 40, pages = 4, logical = 3, width = 8 * 64;
    constexpr int queryWidth = 4 * 64, valueStart = 6 * 64;
    std::vector<float> x(tokens * width), sinks{.4f, -.2f, 2.f, -1.f};
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf(std::sin(float(i) * .013f));
    std::vector<uint16_t> cache(2 * pages * o.pageSize * o.kvHeads * o.headDim, bits(0));
    Device<float> dx(x), ds(sinks), dy(std::vector<float>(tokens * queryWidth));
    Device<uint16_t> dk(cache), dv(cache);
    Device<int> dt(std::vector<int>{2, 0, 3}), dl(std::vector<int>{tokens});
    Device<int> dp(std::vector<int>{0}), da(std::vector<int>{1});
    const void* in[]{dx.p, dk.p, dv.p, dt.p, dl.p, dp.p, da.p, ds.p};
    for (int window : {0, 17}) {
        o.window = window;
        check(RunGptOssAttention(in, dy.p, nullptr, 1, tokens, logical, pages, o, nullptr));
        std::vector<float> expected(tokens * queryWidth);
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < o.qHeads; ++h) {
            const int first = window ? std::max(0, t + 1 - window) : 0;
            double sum = std::exp(double(sinks[h]));
            double accum[64] = {};
            for (int p = first; p <= t; ++p) {
                double score = 0;
                for (int d = 0; d < 64; ++d)
                    score += double(x[t * width + h * 64 + d]) *
                        x[p * width + queryWidth + (h / 2) * 64 + d];
                const double weight = std::exp(score / 8.);
                sum += weight;
                for (int d = 0; d < 64; ++d)
                    accum[d] += weight * x[p * width + valueStart + (h / 2) * 64 + d];
            }
            for (int d = 0; d < 64; ++d)
                expected[t * queryWidth + h * 64 + d] = bf(float(accum[d] / sum));
        }
        compare(dy.read(), expected, .006f,
            window ? "Tiled prefill: paged sliding attention" :
                     "Tiled prefill: paged full attention");
    }
}
void testVocabTop1() {
    for (int width : {7, 65, 100544}) {
      for (int rows : {128,256,512}) {
        std::vector<float> full(size_t(rows) * width * 2), local(size_t(rows) * width);
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < width * 2; ++col)
                full[size_t(row) * width * 2 + col] = bf(((row * 17 + col * 13) % 997 - 498) * .03125f);
        full[width - 1] = full[width] = 32.f; // Cross-shard tie: lower global ID wins.
        full[size_t(width) * 2 + width] = 64.f; // Rank1 wins.
        for (int col = 0; col < width * 2; ++col) {
            full[size_t(2) * width * 2 + col] = std::numeric_limits<float>::quiet_NaN();
            full[size_t(3) * width * 2 + col] = -std::numeric_limits<float>::infinity();
            full[size_t(4) * width * 2 + col] = -0.f;
        }
        full[size_t(5) * width * 2 + 1] = full[size_t(5) * width * 2 + width] = std::numeric_limits<float>::infinity();
        std::vector<float> candidates[2];
        for (int rank = 0; rank < 2; ++rank) {
            for (int row = 0; row < rows; ++row)
                std::copy_n(full.data() + size_t(row) * width * 2 + rank * width,
                            width, local.data() + size_t(row) * width);
            Device<float> input(local), output(std::vector<float>(rows * 2));
            check(RunGptOssVocabTop1(input.p, output.p, rows, width, rank, nullptr));
            candidates[rank] = output.read();
        }
        for (int row = 0; row < rows; ++row) {
            float best = -std::numeric_limits<float>::max(); int index = 0;
            for (int col = 0; col < width * 2; ++col) {
                float value = full[size_t(row) * width * 2 + col];
                if (value > best || (value == best && col < index)) { best = value; index = col; }
            }
            const float a = candidates[0][row * 2], b = candidates[1][row * 2];
            const int ai = int(candidates[0][row * 2 + 1]), bi = int(candidates[1][row * 2 + 1]);
            const bool takeB = b > a || (b == a && bi < ai);
            if ((takeB ? bi : ai) != index || (takeB ? b : a) != best ||
                std::signbit(takeB ? b : a) != std::signbit(best))
                throw std::runtime_error("Compact vocab top1 differs from full vocabulary greedy");
        }
      }
    }
    std::cout << "Compact vocabulary pairs match full greedy, ties/NaN/Inf/signed zero,128/256/512 rows\n";
}
void testLongDecodeAttention(int dimension, int qHeads = 4, int kvHeads = 2) {
    GptOssOptions o; o.kind=1; o.qHeads=qHeads; o.kvHeads=kvHeads; o.headDim=dimension;
    o.layer=1; o.pageSize=16; o.prefill=0;
    const int batch=2, logical=20, pages=40, width=(qHeads+2*kvHeads)*dimension, queryWidth=qHeads*dimension;
    std::vector<float> input(batch*width), sinks(qHeads);
    const float sinkValues[]{12.f,-12.f,2.f,-1.f};
    for (int h=0; h<qHeads; ++h) sinks[h]=sinkValues[h%4];
    for (size_t i=0; i<input.size(); ++i) input[i]=bf(std::sin(float(i)*.13f));
    std::vector<uint16_t> keys(2*pages*16*kvHeads*dimension), values(keys.size());
    for (size_t i=0; i<keys.size(); ++i) {
        keys[i]=bits(std::sin(float(i)*.019f)); values[i]=bits(std::cos(float(i)*.023f));
    }
    std::vector<int> table(batch*logical);
    for (int i=0; i<batch*logical; ++i) table[i]=(i*7)%pages;
    table[3]=-1; // Unmapped pages must contribute neither logits nor values.
    Device<float> dx(input), ds(sinks), dy(std::vector<float>(batch*queryWidth));
    Device<float> scratch(std::vector<float>(size_t(batch) * o.qHeads * 64 * 130));
    Device<uint16_t> dk(keys), dv(values);
    Device<int> dt(table), dl(std::vector<int>{301,301}), dp(std::vector<int>{300,300}), da(std::vector<int>{1,0});
    const void* in[]{dx.p,dk.p,dv.p,dt.p,dl.p,dp.p,da.p,ds.p};
    for (int window : {0,128}) {
        o.window=window;
        check(RunGptOssAttention(in,dy.p,scratch.p,batch,1,logical,pages,o,nullptr));
        auto actualKeys=dk.read(), actualValues=dv.read();
        auto value=[](uint16_t v) { uint32_t bits=uint32_t(v)<<16; float f; std::memcpy(&f,&bits,4); return f; };
        std::vector<float> expected(batch*queryWidth,0);
        for (int h=0; h<qHeads; ++h) {
            double sum=std::exp(double(sinks[h])); std::vector<double> accum(dimension);
            for (int p=window?301-window:0; p<301; ++p) {
                const int page=table[p/16]; if (page<0) continue;
                size_t offset=(((size_t(o.layer)*pages+page)*16+p%16)*kvHeads+h/(qHeads/kvHeads))*dimension;
                double score=0;
                for (int d=0; d<dimension; ++d) score+=double(input[h*dimension+d])*value(actualKeys[offset+d]);
                double weight=std::exp(score/std::sqrt(double(dimension))); sum+=weight;
                for (int d=0; d<dimension; ++d) accum[d]+=weight*value(actualValues[offset+d]);
            }
            for (int d=0; d<dimension; ++d) expected[h*dimension+d]=bf(float(accum[d]/sum));
        }
        compare(dy.read(),expected,.006f,"Split decode attention: long paged KV, sinks, missing page and inactive slot");
        const size_t secondLayer=size_t(pages)*16*kvHeads*dimension;
        for (int p=0; p<logical; ++p) {
            const int page=table[logical+p];
            const size_t offset=secondLayer+size_t(page)*16*kvHeads*dimension;
            for (int i=0; i<16*kvHeads*dimension; ++i)
                if (actualKeys[offset+i]!=keys[offset+i] || actualValues[offset+i]!=values[offset+i])
                    throw std::runtime_error("inactive long-decode cache was modified");
        }
    }
}
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
void testFlashDecode64() {
    const int batch=4,logical=20,pages=batch*logical,dim=64;
    int cases=0;
    for(int heads:{8,32})for(int window:{0,128})for(int length:{-1,0,1,15,16,17,301,320,321}) {
        GptOssOptions o;o.kind=1;o.qHeads=heads;o.kvHeads=heads/8;o.headDim=dim;
        o.layer=1;o.pageSize=16;o.prefill=0;o.window=window;
        const int width=(heads+2*o.kvHeads)*dim;
        std::vector<float> x(batch*width),sinks(heads),expected(batch*heads*dim);
        for(size_t i=0;i<x.size();++i)x[i]=bf(std::sin(float(i)*.13f));
        for(int h=0;h<heads;++h)sinks[h]=h%4==0?12.f:h%4==1?-12.f:h%4==2?2.f:-1.f;
        std::vector<uint16_t> keys(size_t(2)*pages*16*o.kvHeads*dim),values(keys.size());
        for(size_t i=0;i<keys.size();++i){keys[i]=bits(std::sin(float(i)*.019f));values[i]=bits(std::cos(float(i)*.023f));}
        std::vector<int> table(batch*logical),lengths{length,17,301,length},starts(batch,7),active{1,1,1,0};
        for(int b=0;b<batch;++b)for(int p=0;p<logical;++p)
            table[b*logical+p]=p==3?-1:p==7?pages+1:b*logical+logical-p-1;
        // The private adapter receives already-written KV. Independent starts
        // deliberately differ from lengths to catch use of write positions.
        auto unpackBf=[](uint16_t v){uint32_t u=uint32_t(v)<<16;float f;std::memcpy(&f,&u,4);return f;};
        for(int b=0;b<batch;++b)for(int h=0;h<heads;++h) {
            const int end=lengths[b];
            if(!active[b]||end<=0||end>logical*16)continue;
            double sum=std::exp(double(sinks[h]));std::vector<double> accum(dim);
            for(int p=window?std::max(0,end-window):0;p<end;++p) {
                const int page=table[b*logical+p/16];if(page<0||page>=pages)continue;
                const size_t offset=(((size_t(o.layer)*pages+page)*16+p%16)*o.kvHeads+h/8)*dim;
                double score=0;for(int d=0;d<dim;++d)score+=double(x[b*width+h*dim+d])*unpackBf(keys[offset+d]);
                const double weight=std::exp(score/8.);sum+=weight;
                for(int d=0;d<dim;++d)accum[d]+=weight*unpackBf(values[offset+d]);
            }
            for(int d=0;d<dim;++d)expected[(b*heads+h)*dim+d]=bf(float(accum[d]/sum));
        }
        Device<float> dx(x),ds(sinks),dy(expected);
        Device<uint16_t> dk(keys),dv(values);
        Device<int> dt(table),dl(lengths),dp(starts),da(active);
        Device<unsigned char> scratch{std::vector<unsigned char>(GptOssFlashAttentionWorkspace(batch,1,logical,o))};
        const void* in[]{dx.p,dk.p,dv.p,dt.p,dl.p,dp.p,da.p,ds.p};
        check(RunGptOssFlashAttention(in,dy.p,scratch.p,batch,1,logical,pages,o,nullptr));
        compare(dy.read(),expected,.006f,"FlashInfer decode independent FP64 oracle");
        if(dk.read()!=keys||dv.read()!=values)throw std::runtime_error("FlashInfer adapter modified external KV");
        ++cases;
    }
    std::cout<<"FlashInfer decode "<<cases<<" FP64/mask/length/cache cases passed\n";
}
#endif
float unpack(const std::vector<unsigned char>& blocks, const std::vector<unsigned char>& scales, size_t row, int col, int width) {
    unsigned char b = blocks[row * width / 2 + col / 2]; int c = col % 2 ? b >> 4 : b & 15;
    float lut[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return bf(std::ldexp(c & 8 ? -lut[c & 7] : lut[c], int(scales[row * width / 32 + col / 32]) - 127));
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
void testMxfp4Encoding() {
    std::vector<unsigned char> blocks(256 * 16), scales(256);
    for (int e = 0; e < 256; ++e) {
        scales[e] = e;
        for (int byte = 0; byte < 16; ++byte)
            blocks[e * 16 + byte] = ((2 * byte) & 15) | (((2 * byte + 1) & 15) << 4);
    }
    Device<unsigned char> db(blocks), ds(scales);
    Device<float> dy(std::vector<float>(256 * 32));
    check(TestGptOssMxfp4Decode(db.p, ds.p, dy.p, 256));
    auto actual = dy.read();
    for (int e = 0; e < 256; ++e) for (int col = 0; col < 32; ++col) {
        const float value = actual[e * 32 + col];
        if (e == 255) {
            if (!std::isnan(value)) throw std::runtime_error("reserved MXFP4 scale must produce NaN");
        } else {
            const float expected = unpack(blocks, scales, e, col, 32);
            if (value != expected || std::signbit(value) != std::signbit(expected))
                throw std::runtime_error("MXFP4 encoding mismatch, including subnormal/overflow/signed zero");
        }
    }
    std::cout << "All MXFP4 nibbles and E8M0 scales passed exactly\n";
}
#endif
void testPrepackedMarlin(const void* const* inputs,const GptOssOptions& options,int tokens,
    const std::vector<float>& legacy) {
    const int e=options.expertWeightsSharded?(options.experts+1-options.tpRank)/2:options.experts;
    auto pack=[&](int input,bool up,bool scales) {
        const int n=up?2*options.intermediate:options.hidden,k=up?options.hidden:options.intermediate;
        std::vector<unsigned char> original(size_t(e)*n*k/(scales?32:2));
        check(cudaMemcpy(original.data(),inputs[input],original.size(),cudaMemcpyDeviceToHost));
        std::vector<int64_t> shape{e,n,k/32};if(!scales)shape.push_back(16);
        std::vector<unsigned char> output;std::vector<int64_t> packedShape;std::string error;
        if(!PackGptOssMarlinWeight(original.data(),original.size(),shape,
            GptOssMarlinPackSpec{int(up),0,-1,scales},output,packedShape,error))throw std::runtime_error(error);
        return output;
    };
    const auto up=pack(3,true,false),us=pack(4,true,true),down=pack(6,false,false),ds=pack(7,false,true);
    Device<unsigned char> pu(up),pus(us),pd(down),pds(ds);
    // Byte equality against the ACTUAL existing GPU repack kernels.
    for(bool gate:{true,false}) {
        const int k=gate?options.hidden:options.intermediate,n=gate?2*options.intermediate:options.hidden;
        const int pk=(k+(gate?63:127))/(gate?64:128)*(gate?64:128);
        const int pn=(n+(gate?127:63))/(gate?128:64)*(gate?128:64);
        Device<unsigned char> rb{std::vector<unsigned char>(size_t(e)*pk*pn/2)},
            rs{std::vector<unsigned char>(size_t(e)*pk*pn/32)};
        check(TestGptOssMarlinRepack(static_cast<const unsigned char*>(inputs[gate?3:6]),
            static_cast<const unsigned char*>(inputs[gate?4:7]),rb.p,rs.p,e,k,n,pk,pn,nullptr));
        if(rb.read()!=(gate?up:down)||rs.read()!=(gate?us:ds))throw std::runtime_error("Host/actual GPU Marlin bytes differ");
    }
    const void* packedInputs[]{inputs[0],inputs[1],inputs[2],pu.p,pus.p,inputs[5],pd.p,pds.p,inputs[8]};
    auto packedOptions=options;packedOptions.marlinPrepacked=1;
    Device<float> output{std::vector<float>(legacy.size())};
    Device<unsigned char> scratch{std::vector<unsigned char>(GptOssMarlin::Workspace(tokens,packedOptions))};
    {
        GptOssMarlin packed(packedOptions);
        for(int repeat=0;repeat<2;++repeat) {
            check(packed.Run(packedInputs,output.p,scratch.p,tokens,nullptr));
            compare(output.read(),legacy,0.f,"Prepacked versus runtime-repacked Marlin exact output");
        }
        // Compare a direct BF16 combine store against the existing FP32
        // combine followed by the production peer-pack kernel.
        Device<std::uint16_t> expectedWire(std::vector<std::uint16_t>(legacy.size()));
        Device<std::uint16_t> directWire(std::vector<std::uint16_t>(legacy.size()));
        check(GptOssTpPackBf16(output.p,expectedWire.p,legacy.size(),nullptr));
        check(packed.Run(packedInputs,nullptr,scratch.p,tokens,nullptr,directWire.p));
        if(directWire.read()!=expectedWire.read())
            throw std::runtime_error("Direct Marlin BF16 output differs from FP32 combine plus production pack");
    }
    // Context teardown must not free borrowed engine constants.
    if(pu.read()!=up||pus.read()!=us||pd.read()!=down||pds.read()!=ds)
        throw std::runtime_error("Borrowed packed constants changed or were freed");
}
void testMoe(int tokens, int h = 32, int intermediate = 32, bool tiedRouting = false,
             bool batchedDecode = false) {
    GptOssOptions o; o.kind = 2; o.hidden = h; o.intermediate = intermediate; o.experts = 5; o.topK = 2;
    o.prefill = tokens > 8 && !batchedDecode ? 1 : 0;
    std::vector<float> x(tokens * h), router(5 * h), rb{-.2f, .4f, -.1f, .3f, -.5f}, ub(5 * 2 * intermediate), db(5 * h);
    for (size_t i = 0; i < x.size(); ++i) x[i] = bf(std::cos(float(i) * .17f));
    for (size_t i = 0; i < router.size(); ++i) router[i] = bf(std::sin(float(i) * .37f) * .1f);
    if (tiedRouting) {
        std::fill(router.begin(), router.end(), 0.f);
        std::fill(rb.begin(), rb.end(), 0.f);
    }
    for (size_t i = 0; i < ub.size(); ++i) ub[i] = bf(float(int(i % 9) - 4) * .125f);
    for (size_t i = 0; i < db.size(); ++i) db[i] = bf(float(int(i % 7) - 3) * .125f);
    std::vector<unsigned char> up(5 * 2 * intermediate * h / 2), us(5 * 2 * intermediate * h / 32, 123), down(5 * h * intermediate / 2), ds(5 * h * intermediate / 32, 124);
    for (size_t i = 0; i < up.size(); ++i) up[i] = (i * 73 + 11) % 256;
    for (size_t i = 0; i < down.size(); ++i) down[i] = (i * 31 + 57) % 256;
    std::vector<float> expected(tokens * h);
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::pair<float,int>> logits;
        for (int e = 0; e < 5; ++e) { float value = 0; for (int d = 0; d < h; ++d) value += x[t * h + d] * router[e * h + d]; logits.push_back({bf(value + rb[e]),e}); }
        std::sort(logits.begin(), logits.end(), [](auto a, auto b) { return a.first == b.first ? a.second < b.second : a.first > b.first; });
        float denominator = 1 + std::exp(logits[1].first - logits[0].first);
        for (int k = 0; k < 2; ++k) {
            int e = logits[k].second; float probability = bf(std::exp(logits[k].first - logits[0].first) / denominator);
            std::vector<float> hidden(intermediate);
            for (int i = 0; i < intermediate; ++i) {
                int row = e * 2 * intermediate + i * 2; float g = 0, u = 0;
                for (int d = 0; d < h; ++d) { g += x[t * h + d] * unpack(up, us, row, d, h); u += x[t * h + d] * unpack(up, us, row + 1, d, h); }
                g = std::min(bf(bf(g) + ub[row]), 7.f); u = std::clamp(bf(bf(u) + ub[row + 1]), -7.f, 7.f);
                hidden[i] = bf(bf(g * bf(1 / (1 + std::exp(-bf(1.702f * g))))) * bf(u + 1));
            }
            for (int d = 0; d < h; ++d) {
                float v = 0; for (int i = 0; i < intermediate; ++i) v += hidden[i] * unpack(down, ds, e * h + d, i, intermediate);
                expected[t * h + d] += bf(bf(v) + db[e * h + d]) * probability;
            }
        }
    }
    for (float& v : expected) v = bf(v);
    Device<float> dx(x), dr(router), drb(rb), dub(ub), ddb(db), dy(std::vector<float>(expected.size()));
    Device<unsigned char> du(up), dus(us), dd(down), dds(ds), workspace(std::vector<unsigned char>(GptOssMoeWorkspace(tokens,o)));
    const void* in[]{dx.p, dr.p, drb.p, du.p, dus.p, dub.p, dd.p, dds.p, ddb.p};
    check(RunGptOssMoe(in, dy.p, workspace.p, tokens, o, nullptr));
    compare(dy.read(), expected, .002f, "Batched MoE routing, MXFP4, interleaved SwiGLU and biases");
    std::vector<float> shardedSum(expected.size());
    for (int rank = 0; rank < 2; ++rank) {
        GptOssOptions shard = o; shard.tpRank = rank;
        Device<float> partial(std::vector<float>(expected.size()));
        Device<unsigned char> shardWorkspace(std::vector<unsigned char>(GptOssMoeWorkspace(tokens, shard)));
        check(RunGptOssMoe(in, partial.p, shardWorkspace.p, tokens, shard, nullptr));
        const auto values = partial.read();
        for (size_t i = 0; i < values.size(); ++i) shardedSum[i] += values[i];
        shard.expertWeightsSharded = 1;
        Device<unsigned char> lu(shardExpertRows(up, o.experts, rank)),
            lus(shardExpertRows(us, o.experts, rank)),
            ld(shardExpertRows(down, o.experts, rank)),
            lds(shardExpertRows(ds, o.experts, rank));
        Device<float> lub(shardExpertRows(ub, o.experts, rank)),
            ldb(shardExpertRows(db, o.experts, rank));
        const void* localInputs[]{dx.p, dr.p, drb.p, lu.p, lus.p, lub.p, ld.p, lds.p, ldb.p};
        check(RunGptOssMoe(localInputs, partial.p, shardWorkspace.p, tokens, shard, nullptr));
        compare(partial.read(), values, 0.f, "Rank-local original expert constants, warp/grouped paths");
    }
    for (float& value : shardedSum) value = bf(value);
    compare(shardedSum, expected, .002f, "Two-rank expert-parallel MoE sum");
    if(intermediate%64==0) {
        std::vector<float> innerSum(expected.size()),marlinInnerSum(expected.size());
        for(int rank:{0,1}) {
            GptOssOptions local=o;local.intermediate/=2;local.tpRank=-1;local.expertWeightsSharded=0;
            Device<unsigned char> lu(shardIntermediate(up,{o.experts,2*intermediate,h/32,16},rank,1)),
                lus(shardIntermediate(us,{o.experts,2*intermediate,h/32},rank,1)),
                ld(shardIntermediate(down,{o.experts,h,intermediate/32,16},rank,2)),
                lds(shardIntermediate(ds,{o.experts,h,intermediate/32},rank,2));
            Device<float> lub(shardIntermediate(ub,{o.experts,2*intermediate},rank,1)),
                ldb(shardIntermediate(db,{o.experts,h},rank,3)),partial(std::vector<float>(expected.size()));
            const void* localInputs[]{dx.p,dr.p,drb.p,lu.p,lus.p,lub.p,ld.p,lds.p,ldb.p};
            Device<unsigned char> localWorkspace{std::vector<unsigned char>(GptOssMoeWorkspace(tokens,local))};
            check(RunGptOssMoe(localInputs,partial.p,localWorkspace.p,tokens,local,nullptr));
            auto values=partial.read();for(size_t i=0;i<values.size();++i)innerSum[i]+=values[i];
#ifdef GARNET_GPT_OSS_KERNEL_TEST
            if(GptOssMarlin::Workspace(tokens,local)) {
                GptOssMarlin marlin(local);
                Device<unsigned char> scratch{std::vector<unsigned char>(GptOssMarlin::Workspace(tokens,local))};
                check(marlin.Run(localInputs,partial.p,scratch.p,tokens,nullptr));
                values=partial.read();for(size_t i=0;i<values.size();++i)marlinInnerSum[i]+=values[i];
                testPrepackedMarlin(localInputs,local,tokens,values);
            }
#endif
        }
        for(float& v:innerSum)v=bf(v);
        // Partitioning the down reduction introduces BF16 partial rounding.
        // Use the existing compiled TP-versus-unsharded CPU bound; all original
        // single-device and expert-axis kernel bounds remain unchanged.
        compare(innerSum,expected,.025f,"Intermediate TP2 MoE versus independent full CPU oracle (compiled TP bound)");
#ifdef GARNET_GPT_OSS_KERNEL_TEST
        GptOssOptions local=o;local.intermediate/=2;
        if(GptOssMarlin::Workspace(tokens,local)) {
            for(float& v:marlinInnerSum)v=bf(v);
            compare(marlinInnerSum,expected,.025f,"Marlin intermediate TP2 versus independent full CPU oracle (compiled TP bound)");
        }
#endif
    }
#ifdef GARNET_GPT_OSS_KERNEL_TEST
    Device<unsigned char> groupedWorkspace(std::vector<unsigned char>(TestGptOssMoeWorkspace(tokens,o,true)));
    check(TestGptOssMoe(in, dy.p, groupedWorkspace.p, tokens, o, nullptr, true));
    compare(dy.read(), expected, .002f, "Forced grouped MoE, including partial row tiles");
    if (GptOssMarlin::Workspace(tokens, o)) {
        GptOssMarlin marlin(o);
        const size_t marlinBytes = GptOssMarlin::Workspace(tokens, o);
        if (!marlinBytes) throw std::runtime_error("Marlin workspace unexpectedly unsupported");
        Device<unsigned char> marlinWorkspace{std::vector<unsigned char>(marlinBytes)};
        check(marlin.Run(in, dy.p, marlinWorkspace.p, tokens, nullptr));
        compare(dy.read(), expected, .004f, "Marlin MXFP4 expert kernel parity");
        if(tokens>4096) {
            // Independent caller segmentation and a nondefault-stream graph
            // verify offsets, tail, scratch reuse and actual capture lifetime.
            const auto baseline=dy.read();
            Device<float> manual{std::vector<float>(baseline.size())};
            cudaStream_t stream;check(cudaStreamCreate(&stream));
            for(int first=0;first<tokens;first+=4096) {
                const void* segment[9];for(int i=0;i<9;++i)segment[i]=in[i];
                segment[0]=dx.p+size_t(first)*h;
                check(marlin.Run(segment,manual.p+size_t(first)*h,marlinWorkspace.p,
                    std::min(4096,tokens-first),stream));
            }
            check(cudaStreamSynchronize(stream));
            compare(manual.read(),baseline,0.f,"Bounded versus explicit native subcalls exact");
            cudaGraph_t graph;cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            check(marlin.Run(in,manual.p,marlinWorkspace.p,tokens,stream));
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            Device<float> changedGold{std::vector<float>(baseline.size())};
            for(int family=0;family<2;++family) {
                std::vector<float> changed=x;
                if(family)for(size_t i=0;i<changed.size();++i)
                    changed[i]=bf(-.5f*x[i]+float(int(i%7)-3)*.03125f);
                check(cudaMemcpyAsync(dx.p,changed.data(),changed.size()*sizeof(float),
                    cudaMemcpyHostToDevice,stream));
                for(int first=0;first<tokens;first+=4096) {
                    const void* segment[9];for(int i=0;i<9;++i)segment[i]=in[i];
                    segment[0]=dx.p+size_t(first)*h;
                    check(marlin.Run(segment,changedGold.p+size_t(first)*h,marlinWorkspace.p,
                        std::min(4096,tokens-first),stream));
                }
                for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
                check(cudaStreamSynchronize(stream));
                compare(manual.read(),changedGold.read(),0.f,"Bounded graph changing input/scratch reuse exact");
            }
            check(cudaMemcpyAsync(dx.p,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            check(cudaStreamSynchronize(stream));
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
            check(cudaStreamDestroy(stream));
        }
        testPrepackedMarlin(in,o,tokens,dy.read());
        std::vector<float> marlinShardSum(expected.size());
        for (int rank = 0; rank < 2; ++rank) {
            GptOssOptions shard = o; shard.tpRank = rank;
            GptOssMarlin shardMarlin(shard);
            Device<float> partial(std::vector<float>(expected.size()));
            Device<unsigned char> shardWorkspace{
                std::vector<unsigned char>(GptOssMarlin::Workspace(tokens, shard))};
            check(shardMarlin.Run(in, partial.p, shardWorkspace.p, tokens, nullptr));
            const auto values = partial.read();
            for (size_t i = 0; i < values.size(); ++i) marlinShardSum[i] += values[i];
            shard.expertWeightsSharded = 1;
            Device<unsigned char> lu(shardExpertRows(up, o.experts, rank)),
                lus(shardExpertRows(us, o.experts, rank)),
                ld(shardExpertRows(down, o.experts, rank)),
                lds(shardExpertRows(ds, o.experts, rank));
            Device<float> lub(shardExpertRows(ub, o.experts, rank)),
                ldb(shardExpertRows(db, o.experts, rank));
            const void* localInputs[]{dx.p, dr.p, drb.p, lu.p, lus.p, lub.p, ld.p, lds.p, ldb.p};
            GptOssMarlin localMarlin(shard);
            check(localMarlin.Run(localInputs, partial.p, shardWorkspace.p, tokens, nullptr));
            compare(partial.read(), values, 0.f, "Rank-local original expert constants, Marlin path");
            testPrepackedMarlin(localInputs,shard,tokens,partial.read());
        }
        for (float& value : marlinShardSum) value = bf(value);
        compare(marlinShardSum, expected, .004f, "Two-rank Marlin expert-parallel MoE sum");
    }
#endif
}
#ifdef GARNET_GPT_OSS_KERNEL_TEST
void testGqaPrefill64() {
    int cases = 0;
    for (int tokens : {3,21,32,128}) for (int start : {0,509,2005})
        for (int window : {0,128}) for (int heads : {8,32}) {
        GptOssOptions o; o.kind=1; o.qHeads=heads; o.kvHeads=heads/8;
        o.headDim=64; o.pageSize=16; o.layer=1; o.prefill=1; o.window=window;
        constexpr int batch=3;
        const int logical=(start+tokens+15)/16, pages=batch*logical;
        const int width=(heads+2*o.kvHeads)*64, queryWidth=heads*64;
        std::vector<float> x(size_t(batch)*tokens*width), sinks(heads);
        for(size_t i=0;i<x.size();++i)x[i]=bf(std::sin(float(i)*.019f)*1.5f);
        const float sinkValues[]{12.f,-12.f,2.f,-1.f};
        for(int h=0;h<heads;++h)sinks[h]=sinkValues[h%4];
        std::vector<uint16_t> keys(size_t(2)*pages*16*o.kvHeads*64),values(keys.size());
        for(size_t i=0;i<keys.size();++i) {
            keys[i]=bits(std::cos(float(i)*.023f));values[i]=bits(std::sin(float(i)*.017f));
        }
        std::vector<int> table(batch*logical);
        for(int b=0;b<batch;++b)for(int p=0;p<logical;++p)
            table[b*logical+p]=b*logical+(logical-1-p);
        if(logical>3)table[3]=-1;
        if(logical>5)table[5]=pages; // Out-of-range physical page is a hole too.
        const std::vector<int> starts{start,logical*16-3,0},active{1,1,0};
        Device<float> dx(x),ds(sinks),dy{std::vector<float>(size_t(batch)*tokens*queryWidth)};
        Device<uint16_t> dk(keys),dv(values);
        Device<int> dt(table),dp(starts),da(active),dl{std::vector<int>(batch,start+tokens)};
        const void* in[]{dx.p,dk.p,dv.p,dt.p,dl.p,dp.p,da.p,ds.p};
        check(TestGptOssGqaPrefill64(in,dy.p,batch,tokens,logical,pages,o,0,nullptr));
        const auto baseline=dy.read();
        const auto savedK=dk.read(),savedV=dv.read();
        std::vector<float> oracle(baseline.size(),std::numeric_limits<float>::quiet_NaN());
        for(float value:baseline)if(!std::isfinite(value))throw std::runtime_error("Non-finite prefill baseline");
        auto value=[](uint16_t v){uint32_t raw=uint32_t(v)<<16;float f;std::memcpy(&f,&raw,4);return f;};
        // Independent FP64 oracle at first/middle/last queries, every head,
        // including sinks, missing pages, inactive and over-capacity queries.
        for(int b=0;b<batch;++b)for(int t : {0,tokens/2,tokens-1})for(int h=0;h<heads;++h) {
            const int end=starts[b]+t+1;
            double sum=std::exp(double(sinks[h])),accum[64]={};
            if(active[b] && end>0 && end<=logical*16)for(int p=window?std::max(0,end-window):0;p<end;++p) {
                const int page=table[b*logical+p/16];if(page<0||page>=pages)continue;
                const size_t offset=(((size_t(o.layer)*pages+page)*16+p%16)*o.kvHeads+h/8)*64;
                double score=0;
                const size_t row=size_t(b*tokens+t)*width+h*64;
                for(int d=0;d<64;++d)score+=double(x[row+d])*value(savedK[offset+d]);
                const double weight=std::exp(score/8.);sum+=weight;
                for(int d=0;d<64;++d)accum[d]+=weight*value(savedV[offset+d]);
            }
            for(int d=0;d<64;++d) {
                const float expected=bf(float(accum[d]/sum));
                oracle[size_t(b*tokens+t)*queryWidth+h*64+d]=expected;
                if(std::abs(baseline[size_t(b*tokens+t)*queryWidth+h*64+d]-expected)>.006f*(1+std::abs(expected)))
                    throw std::runtime_error("GQA prefill differs from independent FP64 oracle");
            }
        }
        for(int tile : {2,4}) {
            check(TestGptOssGqaPrefill64(in,dy.p,batch,tokens,logical,pages,o,tile,nullptr));
            const auto actual=dy.read();
            if(std::memcmp(actual.data(),baseline.data(),actual.size()*sizeof(float))||dk.read()!=savedK||dv.read()!=savedV)
                throw std::runtime_error("GQA prefill2/4 output/cache differs bitwise from query16 baseline");
        }
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
        Device<unsigned char> flashWorkspace{std::vector<unsigned char>(GptOssFlashAttentionWorkspace(batch,tokens,logical,o))};
        check(RunGptOssFlashAttention(in,dy.p,flashWorkspace.p,batch,tokens,logical,pages,o,nullptr));
        const auto flash=dy.read();
        compare(flash,baseline,.006f,"FlashInfer prefill: full output vs scalar baseline");
        for(size_t i=0;i<flash.size();++i)if(std::isfinite(oracle[i])&&std::abs(flash[i]-oracle[i])>.006f*(1+std::abs(oracle[i])))
            throw std::runtime_error("FlashInfer prefill differs from sampled independent FP64 oracle");
        if(dk.read()!=savedK||dv.read()!=savedV)throw std::runtime_error("FlashInfer prefill mutated KV");
#endif
        const size_t layerBytes=size_t(pages)*16*o.kvHeads*64;
        if(!std::equal(savedK.begin(),savedK.begin()+layerBytes,keys.begin())||
            !std::equal(savedV.begin(),savedV.begin()+layerBytes,values.begin()))
            throw std::runtime_error("GQA prefill modified another cache layer");
        const size_t inactive=layerBytes+size_t(2*logical)*16*o.kvHeads*64;
        if(!std::equal(savedK.begin()+inactive,savedK.end(),keys.begin()+inactive)||
            !std::equal(savedV.begin()+inactive,savedV.end(),values.begin()+inactive))
            throw std::runtime_error("GQA prefill modified inactive cache pages");
        ++cases;
    }
    std::cout << "GQA prefill2/4 exact query16 outputs/cache plus sampled FP64 oracle passed: " << cases << " cases\n";
}
void benchmarkGqaPrefill64(int batch) {
    if(batch<1||batch>128)throw std::runtime_error("Prefill benchmark batch must be1..128");
    constexpr int layers=36,cacheLayers=4,tokens=32,heads=32,kvHeads=4,dim=64,width=40*64;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int start : {224,1952})for(int window : {0,128}) {
        const int logical=(start+tokens+15)/16,pages=batch*logical;
        std::vector<float> x(size_t(layers)*batch*tokens*width),sinks(layers*heads);
        for(size_t i=0;i<x.size();++i)x[i]=bf(float(int(i%31)-15)*.03125f);
        for(size_t i=0;i<sinks.size();++i)sinks[i]=float(int(i%7)-3);
        std::vector<uint16_t> keys(size_t(cacheLayers)*pages*16*kvHeads*dim),values(keys.size());
        for(size_t i=0;i<keys.size();++i){keys[i]=bits(float(int(i%23)-11)*.03125f);values[i]=bits(float(int(i%19)-9)*.03125f);}
        std::vector<int> table(pages);for(int i=0;i<pages;++i)table[i]=i;
        Device<float> dx(x),ds(sinks),dy{std::vector<float>(size_t(layers)*batch*tokens*heads*dim)};
        Device<uint16_t> dk(keys),dv(values);
        Device<int> dt(table),dp{std::vector<int>(batch,start)},da{std::vector<int>(batch,1)},dl{std::vector<int>(batch,start+tokens)};
        std::vector<int> modes{0,2,4};
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
        GptOssOptions flashOptions;flashOptions.prefill=1;flashOptions.headDim=dim;flashOptions.pageSize=16;
        flashOptions.qHeads=heads;flashOptions.kvHeads=kvHeads;
        Device<unsigned char> flashWorkspace{std::vector<unsigned char>(GptOssFlashAttentionWorkspace(batch,tokens,logical,flashOptions))};
        modes.push_back(8); // Diagnostic label8 means FlashInfer, not a scalar query tile.
#endif
        for(int tile : modes) {
            cudaGraph_t graph;cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for(int layer=0;layer<layers;++layer) {
                GptOssOptions o;o.kind=1;o.qHeads=heads;o.kvHeads=kvHeads;o.headDim=dim;
                o.pageSize=16;o.prefill=1;o.window=window;o.layer=layer%cacheLayers;
                const void* in[]{dx.p+size_t(layer)*batch*tokens*width,dk.p,dv.p,dt.p,dl.p,dp.p,da.p,ds.p+layer*heads};
                float* output=dy.p+size_t(layer)*batch*tokens*heads*dim;
                check(TestGptOssGqaPrefill64(in,output,batch,tokens,logical,pages,o,tile==8?-1:tile,stream));
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
                if(tile==8)check(RunGptOssFlashAttention(in,output,flashWorkspace.p,batch,tokens,logical,pages,o,stream));
#endif
            }
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for(int i=0;i<3;++i)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            std::vector<float> samples;
            for(int trial=0;trial<9;++trial) {
                check(cudaEventRecord(begin,stream));
                for(int replay=0;replay<3;++replay)check(cudaGraphLaunch(executable,stream));
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                float ms;check(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms/3);
            }
            std::sort(samples.begin(),samples.end());
            std::cout << "prefill GQA graph batch="<<batch<<" tokens=32 start="<<start<<" window="<<window
                <<" layers=36 cache_layers=4 tile="<<tile<<" median_ms="<<samples[4]<<" min_ms="<<samples.front()<<" max_ms="<<samples.back()<<'\n';
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        }
    }
    check(cudaStreamDestroy(stream));
}
void testBatchRouter() {
    int cases=0;
    for(int tokens : {16,17,65,128,513,4096})for(int width : {96,2880,4096})for(bool ties : {false,true}) {
        GptOssOptions o;o.hidden=width;o.experts=128;o.topK=4;
        std::vector<float> x(size_t(tokens)*width),weight(size_t(o.experts)*width),bias(o.experts);
        for(size_t i=0;i<x.size();++i)x[i]=bf(std::ldexp((i%2?-1.f:1.f)*(.5f+float(i%23)/46.f),int(i%17)-8));
        for(size_t i=0;i<weight.size();++i) {
            const size_t pattern=ties?i%width:i;
            weight[i]=bf(std::ldexp((pattern%3?-1.f:1.f)*(.5f+float(pattern%31)/62.f),int(pattern%11)-5));
        }
        for(int e=0;e<o.experts;++e)bias[e]=ties?0.f:bf(float(e%7-3)*.00390625f);
        Device<float> dx(x),dw(weight),db(bias),logits{std::vector<float>(size_t(tokens)*o.experts)},
            probability{std::vector<float>(tokens*o.topK)};
        Device<int> selected{std::vector<int>(tokens*o.topK)};
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,0,false,nullptr));
        const auto expectedScores=logits.read();
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,0,true,nullptr));
        const auto expectedSelected=selected.read();
        const auto expectedProbabilities=probability.read();
        for(int tile : {2,4}) {
            check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,tile,false,nullptr));
            const auto scores=logits.read();
            if(std::memcmp(scores.data(),expectedScores.data(),scores.size()*sizeof(float)))
                throw std::runtime_error("Tiled batch router scores differ bitwise from scalar256");
            check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,tile,true,nullptr));
            const auto probabilities=probability.read();
            if(selected.read()!=expectedSelected || std::memcmp(probabilities.data(),expectedProbabilities.data(),probabilities.size()*sizeof(float)))
                throw std::runtime_error("Tiled batch router selections/probabilities differ bitwise");
        }
        ++cases;
    }
    std::cout << "Batch router2/4 exact scalar256 scores, top4 and probabilities passed: " << cases << " cases\n";
}
void testTensorCoreBatchRouter() {
    int cases=0,changedTop4Rows=0;float maximum=0;
    for(int tokens : {1,17,127,128,144,448,512,513,4096})for(int width : {96,97,2880,4096})
    for(int experts : {5,65,128})for(bool ties : {false,true}) {
        GptOssOptions o;o.hidden=width;o.experts=experts;o.topK=std::min(4,experts);
        std::vector<float> x(size_t(tokens)*width),weight(size_t(experts)*width),bias(experts);
        for(size_t i=0;i<x.size();++i)x[i]=bf(std::sin(float(i)*.13f)*.3f);
        for(size_t i=0;i<weight.size();++i)weight[i]=ties?0:bf(std::cos(float(i)*.17f)*.07f);
        for(int e=0;e<experts;++e)bias[e]=ties?.125f:bf(float(e%7-3)*.01f);
        Device<float> dx(x),dw(weight),db(bias),logits{std::vector<float>(size_t(tokens)*experts)},
            probability{std::vector<float>(tokens*o.topK)};
        Device<int> selected{std::vector<int>(tokens*o.topK)};
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,0,true,nullptr));
        const auto scalarSelected=selected.read();
        const auto scalarScores=logits.read(),scalarProbability=probability.read();
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,16,false,nullptr));
        const auto scores=logits.read();
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,32,false,nullptr));
        const auto geometryScores=logits.read();
        if(geometryScores.size()!=scores.size() ||
            std::memcmp(geometryScores.data(),scores.data(),scores.size()*sizeof(float)))
            throw std::runtime_error("Two-warp tensor-core router scores differ bitwise from original four-warps");
        // Sample every expert at eight or more rows against an independent
        // FP64 dot product, rather than another GPU reduction implementation.
        for(int t=0;t<tokens;t+=std::max(1,tokens/8))for(int e=0;e<experts;++e) {
            double sum=bias[e];
            for(int k=0;k<width;++k)sum+=double(x[size_t(t)*width+k])*weight[size_t(e)*width+k];
            const float expected=bf(float(sum)),actual=scores[size_t(t)*experts+e];
            maximum=std::max(maximum,std::abs(actual-expected));
            if(!std::isfinite(actual) || std::abs(actual-expected)>.006f*(1+std::abs(expected)))
                throw std::runtime_error("Tensor-core router score differs from independent FP64 oracle");
        }
        std::vector<int> expectedSelected;std::vector<float> expectedProbability;
        for(int t=0;t<tokens;++t) {
            std::vector<int> order(experts);for(int e=0;e<experts;++e)order[e]=e;
            std::sort(order.begin(),order.end(),[&](int a,int b) {
                const float av=scores[size_t(t)*experts+a],bv=scores[size_t(t)*experts+b];
                return av>bv || (av==bv && a<b);
            });
            bool changed=false;float total=0;
            for(int k=0;k<o.topK;++k)total+=std::exp(scores[size_t(t)*experts+order[k]]-scores[size_t(t)*experts+order[0]]);
            for(int k=0;k<o.topK;++k) {
                changed|=order[k]!=scalarSelected[t*o.topK+k];
                expectedSelected.push_back(order[k]);
                expectedProbability.push_back(bf(std::exp(scores[size_t(t)*experts+order[k]]-scores[size_t(t)*experts+order[0]])/total));
            }
            changedTop4Rows+=changed;
        }
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,16,true,nullptr));
        if(selected.read()!=expectedSelected)throw std::runtime_error("Tensor-core router top-K differs from independent score sorting");
        compare(probability.read(),expectedProbability,.006f,"Tensor-core router probabilities vs independent CPU");
        const auto tensorCoreProbability=probability.read();
        const auto tensorCorePostTopK=logits.read();
        check(TestGptOssBatchRouter(dx.p,dw.p,db.p,logits.p,selected.p,probability.p,tokens,o,32,true,nullptr));
        const auto geometryProbability=probability.read(),geometryPostTopK=logits.read();
        if(selected.read()!=expectedSelected ||
            geometryProbability.size()!=tensorCoreProbability.size() ||
            geometryPostTopK.size()!=tensorCorePostTopK.size() ||
            std::memcmp(geometryProbability.data(),tensorCoreProbability.data(),tensorCoreProbability.size()*sizeof(float)) ||
            std::memcmp(geometryPostTopK.data(),tensorCorePostTopK.data(),tensorCorePostTopK.size()*sizeof(float)))
            throw std::runtime_error("Two-warp tensor-core router top-K/probabilities/post-scores differ bitwise from original");
        const char* flag=std::getenv("GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE");
        if(tokens>=1024 && flag && flag[0]=='1' && flag[1]=='\0') {
            o.prefill=1;
            const void* in[]{dx.p,dw.p,db.p};
            check(RunGptOssMoeRoute(in,selected.p,probability.p,logits.p,tokens,o,nullptr));
            if(selected.read()!=expectedSelected)throw std::runtime_error("Opt-in production prefill router differs from tensor-core score sorting");
            compare(probability.read(),expectedProbability,.006f,"Production prefill router probabilities vs independent CPU");
        }
        if(tokens>=127 && tokens<=513) {
            o.prefill=0;
            const char* decodeFlag=std::getenv("GARNET_GPT_OSS_DECODE_ROUTER_TENSORCORE");
            const bool enabled=decodeFlag && decodeFlag[0]=='1' && decodeFlag[1]=='\0' &&
                GptOssDecodeTensorCoreRouterSupported(o.prefill,tokens,width,experts,o.topK);
            const void* in[]{dx.p,dw.p,db.p};
            check(RunGptOssMoeRoute(in,selected.p,probability.p,logits.p,tokens,o,nullptr));
            const auto observedScores=logits.read(),observedProbability=probability.read();
            const auto& wantedScores=enabled?tensorCorePostTopK:scalarScores;
            const auto& wantedProbability=enabled?tensorCoreProbability:scalarProbability;
            if(selected.read()!=(enabled?expectedSelected:scalarSelected) ||
                std::memcmp(observedScores.data(),wantedScores.data(),wantedScores.size()*sizeof(float)) ||
                std::memcmp(observedProbability.data(),wantedProbability.data(),wantedProbability.size()*sizeof(float)))
                throw std::runtime_error("Production decode router dispatch differs from selected native reference");
        }
        ++cases;
    }
    std::cout<<"Tensor-core BF16 router FP64 scores/CPU sorting/probabilities passed: "<<cases
             <<" cases; maximum score error="<<maximum<<" changed scalar top4 rows="<<changedTop4Rows<<'\n';
}
void benchmarkBatchRouter() {
    constexpr int layers=36,width=2880,experts=128;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int tokens : {128,144,256,448,512,4096}) {
        std::vector<float> x(size_t(layers)*tokens*width),weights(size_t(layers)*experts*width),bias(layers*experts);
        for(size_t i=0;i<x.size();++i)x[i]=bf(float(int(i%23)-11)*.015625f);
        for(size_t i=0;i<weights.size();++i)weights[i]=bf(float(int(i%31)-15)*.0078125f);
        Device<float> dx(x),dw(weights),db(bias),logits{std::vector<float>(size_t(layers)*tokens*experts)};
        GptOssOptions o;o.hidden=width;o.experts=experts;o.topK=4;
        for(int tile : {0,2,4,16}) {
            cudaGraph_t graph;cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for(int layer=0;layer<layers;++layer)check(TestGptOssBatchRouter(
                dx.p+size_t(layer)*tokens*width,dw.p+size_t(layer)*experts*width,db.p+layer*experts,
                logits.p+size_t(layer)*tokens*experts,nullptr,nullptr,tokens,o,tile,false,stream));
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for(int i=0;i<3;++i)check(cudaGraphLaunch(executable,stream));
            check(cudaStreamSynchronize(stream));
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            std::vector<float> samples;
            for(int trial=0;trial<9;++trial) {
                check(cudaEventRecord(begin,stream));
                for(int replay=0;replay<3;++replay)check(cudaGraphLaunch(executable,stream));
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                float ms;check(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms/3);
            }
            std::sort(samples.begin(),samples.end());
            std::cout << "router score graph tokens=" << tokens << " width=2880 experts=128 layers=36 tile="
                << tile << " median_ms=" << samples[4] << " min_ms=" << samples.front() << " max_ms=" << samples.back() << '\n';
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        }
    }
    check(cudaStreamDestroy(stream));
}
std::vector<int> metadataRoutes(int slots,int experts,int pattern,int layer=0) {
    std::vector<int> selected(slots);
    for(int s=0;s<slots;++s)selected[s]=pattern==0 ? (s%4+layer)%experts
        : ((s*17+s/11+layer*7)%experts);
    return selected;
}
void testMarlinMetadata() {
    int cases=0;
    for(int slots : {1,63,68,512,2052,16384})for(int count : {1,5,128,256})
    for(int rank : {-1,0,1})for(int block : {8,32,64})for(int pattern : {0,1}) {
        const auto selected=metadataRoutes(slots,count,pattern);
        const int capacity=slots+count*block,blocks=(capacity+block-1)/block;
        std::vector<int> expectedSorted(capacity,-99),expectedExperts(blocks,-99);
        int total=0;
        const int local=rank<0?count:(count+1-rank)/2;
        for(int e=0;e<local;++e) {
            const int global=rank<0?e:2*e+rank,start=total;
            for(int s=0;s<slots;++s)if(selected[s]==global)expectedSorted[total++]=s;
            const int used=total-start,padded=(used+block-1)/block*block;
            while(total<start+padded)expectedSorted[total++]=slots;
            for(int row=0;row<used;row+=block)expectedExperts[(start+row)/block]=e;
        }
        Device<int> input(selected),scratch{std::vector<int>(count*2)};
        for(bool parallel : {false,true}) {
            Device<int> sorted{std::vector<int>(capacity,-99)},experts{std::vector<int>(blocks,-99)},padded{std::vector<int>(1,-99)};
            check(TestGptOssMarlinMetadata(input.p,sorted.p,experts.p,padded.p,scratch.p,
                slots,count,rank,block,parallel,nullptr));
            if(sorted.read()!=expectedSorted || experts.read()!=expectedExperts || padded.read()[0]!=total)
                throw std::runtime_error("Marlin stable metadata differs from integer CPU oracle");
        }
        ++cases;
    }
    std::cout << "Marlin stable metadata exact CPU/legacy/parallel parity passed: " << cases << " cases\n";
}
void benchmarkMarlinMetadata() {
    constexpr int layers=36,expertCount=128,block=32;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int slots : {512,16384})for(int pattern : {0,1}) {
        const int capacity=slots+expertCount*block,blocks=(capacity+block-1)/block;
        std::vector<int> routes;
        for(int layer=0;layer<layers;++layer) {
            const auto values=metadataRoutes(slots,expertCount,pattern,layer);
            routes.insert(routes.end(),values.begin(),values.end());
        }
        Device<int> selected(routes),sorted{std::vector<int>(layers*capacity)},
            experts{std::vector<int>(layers*blocks)},padded{std::vector<int>(layers)},
            scratch{std::vector<int>(layers*expertCount*2)};
        for(bool parallel : {false,true}) {
            cudaGraph_t graph;cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for(int layer=0;layer<layers;++layer)check(TestGptOssMarlinMetadata(
                selected.p+layer*slots,sorted.p+layer*capacity,experts.p+layer*blocks,
                padded.p+layer,scratch.p+layer*expertCount*2,slots,expertCount,0,block,parallel,stream));
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for(int i=0;i<5;++i)check(cudaGraphLaunch(executable,stream));
            check(cudaStreamSynchronize(stream));
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            std::vector<float> samples;
            for(int trial=0;trial<15;++trial) {
                check(cudaEventRecord(begin,stream));
                for(int replay=0;replay<10;++replay)check(cudaGraphLaunch(executable,stream));
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                float ms;check(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms/10);
            }
            std::sort(samples.begin(),samples.end());
            std::cout << "metadata graph slots=" << slots << " pattern=" << pattern
                << " layers=" << layers << " rank=0 experts=128 block=32 parallel=" << parallel
                << " median_ms=" << samples[7] << " min_ms=" << samples.front()
                << " max_ms=" << samples.back() << '\n';
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        }
    }
    check(cudaStreamDestroy(stream));
}
#endif
int main(int argc, char** argv) { try {
    if(argc==2 && std::strcmp(argv[1],"--invalid-router-warps")==0) {
        const char* flag=std::getenv("GARNET_GPT_OSS_TENSOR_ROUTER_EXPERT_WARPS");
        if(!flag || std::strcmp(flag,"2")==0 || std::strcmp(flag,"4")==0)
            throw std::runtime_error("Negative router fixture requires malformed policy");
        GptOssOptions o;o.hidden=96;o.experts=8;o.topK=4;
        if(TestGptOssBatchRouter(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,128,o,32,false,nullptr)!=cudaErrorInvalidValue ||
            RunGptOssMoeRoute(nullptr,nullptr,nullptr,nullptr,128,o,nullptr)!=cudaErrorInvalidValue)
            throw std::runtime_error("Malformed tensor router geometry must reject before pointer access/router work");
        std::cout << "Invalid router warp policy rejected before router work PASS\n";
        return 0;
    }
    if(argc==2&&std::strcmp(argv[1],"--hybrid-kv-attention-parity")==0) {
#ifndef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
        throw std::runtime_error("Hybrid KV gate requires compiled FlashInfer adapter");
#else
        testHybridRingAttention64();check(cudaDeviceSynchronize());return 0;
#endif
    }
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
    if(argc==2&&std::strcmp(argv[1],"--flash-decode-parity")==0){testFlashDecode64();return 0;}
#endif
#ifdef GARNET_GPT_OSS_KERNEL_TEST
    if(argc==2 && std::strcmp(argv[1],"--tensorcore-router-parity")==0) {
        testBatchRouter();testTensorCoreBatchRouter();return 0;
    }
    if(argc==2 && std::strcmp(argv[1],"--prefill-gqa-parity")==0) {
        testGqaPrefill64();return 0;
    }
    if((argc==2 || argc==3) && std::strcmp(argv[1],"--prefill-gqa-benchmark")==0) {
        testGqaPrefill64();benchmarkGqaPrefill64(argc==3?std::stoi(argv[2]):8);return 0;
    }
    if(argc==2 && std::strcmp(argv[1],"--router-benchmark")==0) {
        testBatchRouter();benchmarkBatchRouter();return 0;
    }
    if(argc==2 && std::strcmp(argv[1],"--metadata-benchmark")==0) {
        testMarlinMetadata();benchmarkMarlinMetadata();return 0;
    }
#endif
    if (argc == 2 && std::strcmp(argv[1], "--workspace") == 0) {
        GptOssOptions options;
        options.kind = 2; options.hidden = options.intermediate = 2880;
        options.experts = 128; options.topK = 8; options.tpRank = 0;
        for (int rows : {1, 8, 32, 128, 513, 1023, 1024, 4096, 4097, 4608, 7168, 8020, 8192, 8193}) {
            options.prefill = 1;
            const auto prefillBytes = GptOssMarlin::Workspace(rows, options);
            options.prefill = 0;
            std::cout << rows << ' ' << GptOssMoeWorkspace(rows, options) << ' '
                << prefillBytes << ' ' << GptOssMarlin::Workspace(rows, options) << '\n';
        }
        return 0;
    }
    if(argc==2 && std::strcmp(argv[1],"--prefill-down-grid-rejection")==0) {
        if(!std::getenv("GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K") ||
           std::strcmp(std::getenv("GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_K"),"64")!=0 ||
           !std::getenv("GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM") ||
           std::strcmp(std::getenv("GARNET_GPT_OSS_MARLIN_PREFILL_DOWN_CTAS_PER_SM"),"4")!=0)return 2;
        GptOssOptions probe;probe.kind=2;probe.hidden=96;probe.intermediate=256;
        probe.experts=5;probe.topK=2;probe.prefill=1;
        GptOssMarlin marlin(probe);
        // No buffers exist: rejection must happen before pointer access/refit.
        for(int rows:{1024,4096})for(int repeat=0;repeat<2;++repeat)
            if(marlin.Run(nullptr,nullptr,nullptr,rows,nullptr)!=cudaErrorInvalidConfiguration)
                throw std::runtime_error("Unsafe down-K64 grid was not rejected before memory access");
        check(cudaDeviceSynchronize());
        std::cout<<"Down-K64 CTA4 rejected before pointer access/repacking/launch, cold/cached PASS\n";
        return 0;
    }
    if(argc==3 && (std::strcmp(argv[1],"--bounded-prefill-parity")==0 ||
                   std::strcmp(argv[1],"--bounded-prefill-tp-sized-parity")==0)) {
        const int rows=std::stoi(argv[2]);
        if(rows!=4097 && rows!=4608 && rows!=7168 && rows!=8020 && rows!=8192)return 2;
        GptOssOptions probe;probe.kind=2;probe.hidden=96;probe.intermediate=256;
        probe.experts=5;probe.topK=2;probe.prefill=1;
        if(!GptOssMarlin::Workspace(rows,probe) ||
            GptOssMarlin::Workspace(rows,probe)!=GptOssMarlin::Workspace(4096,probe))
            throw std::runtime_error("Bounded gate requires4096-row subcall workspace");
        probe.prefill=0;
        if(GptOssMarlin::Workspace(rows,probe))
            throw std::runtime_error("Bounded prefill changed decode support");
        testMoe(rows,96,std::strcmp(argv[1],"--bounded-prefill-tp-sized-parity")==0?2880:256,false);
        check(cudaDeviceSynchronize());return 0;
    }
    if(argc==2 && std::strcmp(argv[1],"--tp-sized-prefill-down-parity")==0) {
        GptOssOptions probe;probe.kind=2;probe.hidden=96;probe.intermediate=2880;
        probe.experts=5;probe.topK=2;probe.prefill=1;
        if(!GptOssMarlin::Workspace(1024,probe))
            throw std::runtime_error("Representative TP gate requires enabled large-row Marlin support");
        testMoe(1024,96,2880,false);
        check(cudaDeviceSynchronize());return 0;
    }
    if (argc == 2 && (std::strcmp(argv[1], "--large-prefill-parity") == 0 ||
                     std::strcmp(argv[1], "--prefill-down-parity") == 0)) {
        GptOssOptions probe; probe.kind=2; probe.hidden=probe.intermediate=32;
        probe.experts=5; probe.topK=2; probe.prefill=1;
        if(!GptOssMarlin::Workspace(4096,probe))
            throw std::runtime_error("Large-prefill gate requires enabled 4096-row Marlin support");
        // Independent full CPU oracle plus rank-local expert/intermediate
        // shards and exact original/prepacked comparison at dispatch bounds.
        testMoe(1023,32,32,true);
        testMoe(1024,96,64,false);
        testMoe(4096,32,32,true);
        testMoe(1024,32,32,true,true); // Decode must retain its old tile.
        if(std::strcmp(argv[1], "--prefill-down-parity") == 0)
            testMoe(1024,96,256,false); // Multiple real K64 reduction tiles, both TP axes.
        check(cudaDeviceSynchronize()); return 0;
    }
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
    testFlashDecode64();
#endif
#ifdef GARNET_GPT_OSS_KERNEL_TEST
    testGqaPrefill64();
    testBatchRouter();
    testTensorCoreBatchRouter();
    testMarlinMetadata();
    testMxfp4Encoding();
#endif
        testDecodeGemv(2560, 2880); testDecodeGemv(2880, 2048);
        for (int rank : {0, 1}) {
            testDecodeGemvSharded(true, rank);
            testDecodeGemvSharded(false, rank);
        }
        testVocabTop1(); testRmsNorm(); testRope(); for (int dimension : {8, 64, 128}) { testAttention(dimension); testLongDecodeAttention(dimension); } testLongDecodeAttention(64, 8, 1); testLongDecodeAttention(64, 32, 4); testLongPrefillAttention64(); for (int tokens : {1, 3, 17, 65}) { testMoe(tokens); testMoe(tokens, 96, 64); } testMoe(65, 96, 64, true); testMoe(513, 32, 32, true); testMoe(128, 96, 64, false, true); testMoe(3,96,2880); testMoe(513,96,64,true); check(cudaDeviceSynchronize()); return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; } }
