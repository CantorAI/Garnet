// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_kernels.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <algorithm>
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
    Device<uint16_t> dk(keys), dv(values);
    Device<int> dt(table), dl(lengths), dp(starts), da(active);
    const void* in[]{dx.p, dk.p, dv.p, dt.p, dl.p, dp.p, da.p, ds.p};
    for (int window : {0, 2}) {
        o.window = window;
        check(RunGptOssAttention(in, dy.p, batch, tokens, logical, pages, o, nullptr));
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
    check(RunGptOssAttention(decode, dout.p, batch, 1, logical, pages, o, nullptr));
    auto actual = dout.read(); std::vector<float> expected(batch * queryWidth, 0);
    std::copy_n(prefill.begin() + 2 * queryWidth, queryWidth, expected.begin());
    compare(actual, expected, .006f, "Cached decode + inactive slot");
    if (dk.read() != savedK || dv.read() != savedV) throw std::runtime_error("decode cache mutation mismatch");
    for (size_t i = 0; i < keys.size() / 2; ++i) if (savedK[i] != keys[i]) throw std::runtime_error("wrong KV layer modified");
}
float unpack(const std::vector<unsigned char>& blocks, const std::vector<unsigned char>& scales, size_t row, int col, int width) {
    unsigned char b = blocks[row * width / 2 + col / 2]; int c = col % 2 ? b >> 4 : b & 15;
    float lut[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return bf(std::ldexp(c & 8 ? -lut[c & 7] : lut[c], int(scales[row * width / 32 + col / 32]) - 127));
}
void testMoe(int tokens, int h = 32, int intermediate = 32, bool tiedRouting = false) {
    GptOssOptions o; o.kind = 2; o.hidden = h; o.intermediate = intermediate; o.experts = 5; o.topK = 2;
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
}
int main() { try { testRope(); for (int dimension : {8, 64, 128}) testAttention(dimension); for (int tokens : {1, 3, 17, 65}) { testMoe(tokens); testMoe(tokens, 96, 64); } testMoe(65, 96, 64, true); check(cudaDeviceSynchronize()); return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; } }
