// SPDX-License-Identifier: Apache-2.0
// Independent native gate for the private, unselected XQA adapter. NDEBUG
// never disables checks. No TensorRT/plugin/model/performance claim follows.
#include "gpt_oss_xqa_attention.h"
#include "gpt_oss_xqa_layout.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace Garnet;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void cleanup(cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        std::cerr << "XQA cleanup failed: " << cudaGetErrorString(status) << '\n';
        std::abort();
    }
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    std::size_t count;
    explicit DeviceBuffer(std::size_t count) : count(count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "buffer allocation");
    }
    ~DeviceBuffer() { cleanup(cudaFree(data)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& source, cudaStream_t stream) {
        require(source.size() == count, "upload extent");
        check(cudaMemcpyAsync(data, source.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    }
    std::vector<T> read(cudaStream_t stream) const {
        std::vector<T> result(count);
        check(cudaMemcpyAsync(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "read");
        check(cudaStreamSynchronize(stream), "read completion");
        return result;
    }
};
struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "stream creation"); }
    ~Stream() { cleanup(cudaStreamSynchronize(value)); cleanup(cudaStreamDestroy(value)); }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaStream_t stream;
    explicit Graph(cudaStream_t stream) : stream(stream) {}
    ~Graph() {
        cleanup(cudaStreamSynchronize(stream));
        if (executable) cleanup(cudaGraphExecDestroy(executable));
        if (graph) cleanup(cudaGraphDestroy(graph));
    }
};
float rounded(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    bits &= 0xffff0000U;
    std::memcpy(&value, &bits, 4);
    return value;
}
__host__ __device__ float cacheValue(std::size_t index, unsigned seed) {
    // Exactly BF16 representable, independent of conversion/kernel arithmetic.
    const int value = int((index * 17 + seed * 13) % 97) - 48;
    return value / 256.f;
}
__global__ void fillCache(__nv_bfloat16* cache, std::size_t count, unsigned seed) {
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < count; index += std::size_t(gridDim.x) * blockDim.x)
        cache[index] = __float2bfloat16(cacheValue(index, seed));
}
struct Spec {
    int batch, logical, window, kvHeads, layer;
    int pattern; // 0 mixed, 1 dense, 2 holes, 3 window-excluded holes, 4 all holes
    bool uniquePages = false;
    int uniformLength = 0; // 0 selects mixed lengths
};
struct Inputs {
    std::vector<float> query, sinks;
    std::vector<int> table, lengths, starts, active;
};
Inputs makeInputs(const Spec& spec, int physical, int revision) {
    const int qHeads = spec.kvHeads * 8, packed = (qHeads + 2 * spec.kvHeads) * 64;
    Inputs in;
    in.query.resize(std::size_t(spec.batch) * packed);
    in.sinks.resize(qHeads); in.table.resize(std::size_t(spec.batch) * spec.logical);
    in.lengths.resize(spec.batch); in.starts.resize(spec.batch); in.active.resize(spec.batch);
    const std::array<int, 14> ends{1, 15, 16, 17, 127, 128, 129,
        std::min(2005, spec.logical * 16), spec.logical * 16 - 1,
        spec.logical * 16, 0, -1, spec.logical * 16 + 1, INT_MAX};
    for (int b = 0; b < spec.batch; ++b) {
        int end = spec.uniformLength ? spec.uniformLength : ends[(b + revision * 3) % ends.size()];
        if (!spec.uniformLength && b % 29 == 28) end = INT_MIN;
        in.lengths[b] = end;
        // Decode must use lengths, not starts; deliberately unrelated values.
        in.starts[b] = b % 2 ? INT_MIN : INT_MAX;
        in.active[b] = b % 19 == 18 ? 0 : (b % 5 ? 1 : -7);
        for (int p = 0; p < spec.logical; ++p) {
            int page = spec.uniquePages ? b * spec.logical + p
                : (p * 3 + b * 7 + revision) % physical;
            const int pattern = spec.pattern == 0 ? b % 5 : spec.pattern;
            if (pattern == 2 && p % 3 == (revision % 3))
                page = p % 2 ? -1 : physical;
            if (pattern == 3 && spec.window && end > spec.window + 16 &&
                    p * 16 + 15 < end - spec.window)
                page = p % 2 ? INT_MIN : INT_MAX;
            if (pattern == 4) page = -1;
            // Nonvisible tail entries must also be sanitized.
            if (p == spec.logical - 1 && end > 0 && end < p * 16)
                page = INT_MAX;
            in.table[std::size_t(b) * spec.logical + p] = page;
        }
        for (int d = 0; d < packed; ++d) {
            const int tagged = int((std::size_t(b) * packed + d * 7 + revision * 31) % 113) - 56;
            in.query[std::size_t(b) * packed + d] = tagged / 128.f + (revision ? .000031f : 0.f);
        }
    }
    for (int h = 0; h < qHeads; ++h)
        in.sinks[h] = std::array<float, 4>{-6.f, .37f, 3.f, -.03f}[(h + revision) % 4];
    return in;
}
bool validRow(const Spec& spec, const Inputs& in, int b) {
    return in.active[b] != 0 && in.lengths[b] > 0 && in.lengths[b] <= spec.logical * 16;
}
int modeOracle(const Spec& spec, const Inputs& in, int physical, int b) {
    if (!validRow(spec, in, b)) return 0;
    // Token-distance enumeration intentionally does not import page classifier.
    for (int pos = 0; pos < in.lengths[b]; ++pos) {
        if (spec.window && in.lengths[b] - pos > spec.window) continue;
        const int page = in.table[std::size_t(b) * spec.logical + pos / 16];
        if (page < 0 || page >= physical) return 2;
    }
    return 1;
}
std::array<float, 64> attentionOracle(const Spec& spec, const Inputs& in,
    int physical, int b, int head, std::size_t layerElements, unsigned keySeed, unsigned valueSeed) {
    std::array<float, 64> result{};
    if (!validRow(spec, in, b)) return result;
    struct Token { std::size_t offset; double score; };
    std::vector<Token> tokens;
    const int qHeads = spec.kvHeads * 8, packed = (qHeads + 2 * spec.kvHeads) * 64;
    double maximum = in.sinks[head];
    for (int pos = 0; pos < in.lengths[b]; ++pos) {
        if (spec.window && in.lengths[b] - pos > spec.window) continue;
        const int page = in.table[std::size_t(b) * spec.logical + pos / 16];
        if (page < 0 || page >= physical) continue;
        const auto offset = std::size_t(spec.layer) * layerElements +
            ((std::size_t(page) * 16 + pos % 16) * spec.kvHeads + head / 8) * 64;
        double score = 0;
        for (int d = 0; d < 64; ++d)
            score += double(in.query[std::size_t(b) * packed + head * 64 + d]) *
                double(cacheValue(offset + d, keySeed));
        score /= 8.;
        tokens.push_back({offset, score}); maximum = std::max(maximum, score);
    }
    double denominator = std::exp(double(in.sinks[head]) - maximum);
    std::array<double, 64> accumulator{};
    for (const auto& token : tokens) {
        const double weight = std::exp(token.score - maximum);
        denominator += weight;
        for (int d = 0; d < 64; ++d)
            accumulator[d] += weight * double(cacheValue(token.offset + d, valueSeed));
    }
    for (int d = 0; d < 64; ++d) result[d] = rounded(float(accumulator[d] / denominator));
    return result;
}
struct Counts {
    std::size_t cases = 0, invocations = 0, rows = 0, tables = 0, checked = 0, zeros = 0, guards = 0;
    std::array<std::size_t, 3> modes{};
    double maxAbs = 0;
};
void guardChecks(const void* const* inputs, float* y, void* workspace,
    const Spec& spec, int physical, GptOssOptions options,
    const GptOssXqaContextState& state, cudaStream_t stream, Counts& counts) {
    auto run = [&](const void* const* in, float* out, void* scratch, int tokens,
                   int pages, int physicalPages, GptOssOptions o, GptOssXqaContextState s) {
        ++counts.guards;
        require(RunGptOssXqaAttention(in, out, scratch, spec.batch, tokens,
            pages, physicalPages, o, s, stream) != cudaSuccess, "invalid argument enqueued");
    };
    run(inputs, y, workspace, 1, spec.logical, physical, options, {});
    auto wrongDevice = state; wrongDevice.device = state.device + 1;
    run(inputs, y, workspace, 1, spec.logical, physical, options, wrongDevice);
    run(nullptr, y, workspace, 1, spec.logical, physical, options, state);
    run(inputs, nullptr, workspace, 1, spec.logical, physical, options, state);
    run(inputs, y, nullptr, 1, spec.logical, physical, options, state);
    run(inputs, y, static_cast<char*>(workspace) + 1, 1, spec.logical, physical, options, state);
    run(inputs, reinterpret_cast<float*>(reinterpret_cast<char*>(y) + 1), workspace,
        1, spec.logical, physical, options, state);
    for (int i = 0; i < 8; ++i) {
        std::array<const void*, 8> changed{}; std::copy(inputs, inputs + 8, changed.begin());
        changed[i] = nullptr;
        run(changed.data(), y, workspace, 1, spec.logical, physical, options, state);
        changed[i] = static_cast<const char*>(inputs[i]) + 1;
        run(changed.data(), y, workspace, 1, spec.logical, physical, options, state);
    }
    run(inputs, y, workspace, 2, spec.logical, physical, options, state);
    run(inputs, y, workspace, 1, 257, physical, options, state);
    run(inputs, y, workspace, 1, spec.logical, 0, options, state);
    auto invalid = options; invalid.prefill = 1;
    run(inputs, y, workspace, 1, spec.logical, physical, invalid, state);
    invalid = options; invalid.headDim = 128;
    run(inputs, y, workspace, 1, spec.logical, physical, invalid, state);
    invalid = options; invalid.layer = -1;
    run(inputs, y, workspace, 1, spec.logical, physical, invalid, state);
    invalid = options; invalid.kind = 0;
    run(inputs, y, workspace, 1, spec.logical, physical, invalid, state);
    invalid = options; invalid.layer = INT_MAX;
    run(inputs, y, workspace, 1, spec.logical, INT_MAX, invalid, state);
    check(cudaPeekAtLastError(), "invalid requests leave CUDA error state clean");
}
void runCase(const Spec& spec, int caseIndex, Counts& counts) {
    const int qHeads = spec.kvHeads * 8;
    const int physical = spec.uniquePages ? spec.batch * spec.logical : 9;
    const std::size_t perLayer = std::size_t(physical) * 16 * spec.kvHeads * 64;
    const std::size_t cacheElements = perLayer * 2;
    const unsigned keySeed = 53 + caseIndex, valueSeed = 91 + caseIndex;
    GptOssOptions options; options.kind = 1; options.qHeads = qHeads;
    options.kvHeads = spec.kvHeads; options.headDim = 64;
    options.layer = spec.layer; options.window = spec.window;
    GptOssXqaContextState state;
    check(InitializeGptOssXqaContext(&state), "explicit context initialization");
    require(state.device >= 0 && state.dynamicSharedBytes > 0, "context initialized");
    Stream stream;
    DeviceBuffer<unsigned short> keys(cacheElements), values(cacheElements);
    DeviceBuffer<float> query(std::size_t(spec.batch) * (qHeads + 2 * spec.kvHeads) * 64), sinks(qHeads);
    DeviceBuffer<int> table(std::size_t(spec.batch) * spec.logical), lengths(spec.batch), starts(spec.batch), active(spec.batch);
    DeviceBuffer<float> output(std::size_t(spec.batch) * qHeads * 64);
    const auto bytes = GptOssXqaAttentionWorkspace(spec.batch, 1, spec.logical, options);
    require(bytes > 0, "supported workspace");
    DeviceBuffer<unsigned char> workspace(bytes);
    fillCache<<<std::min<std::size_t>((cacheElements + 255) / 256, 4096), 256, 0, stream.value>>>(
        reinterpret_cast<__nv_bfloat16*>(keys.data), cacheElements, keySeed);
    check(cudaGetLastError(), "key cache initialization");
    fillCache<<<std::min<std::size_t>((cacheElements + 255) / 256, 4096), 256, 0, stream.value>>>(
        reinterpret_cast<__nv_bfloat16*>(values.data), cacheElements, valueSeed);
    check(cudaGetLastError(), "value cache initialization");
    const auto initialKeys = keys.read(stream.value), initialValues = values.read(stream.value);
    // Verify the generated cache independently before using it as an oracle.
    for (std::size_t i = 0; i < cacheElements; ++i) {
        uint32_t k, v; float kf = cacheValue(i, keySeed), vf = cacheValue(i, valueSeed);
        std::memcpy(&k, &kf, 4); std::memcpy(&v, &vf, 4);
        require(initialKeys[i] == (k >> 16) && initialValues[i] == (v >> 16), "initial BF16 cache bytes");
    }
    const void* inputs[]{query.data, keys.data, values.data, table.data,
        lengths.data, starts.data, active.data, sinks.data};
    guardChecks(inputs, output.data, workspace.data, spec, physical, options, state, stream.value, counts);
    auto invoke = [&] {
        check(RunGptOssXqaAttention(inputs, output.data, workspace.data,
            spec.batch, 1, spec.logical, physical, options, state, stream.value), "native attention enqueue");
    };
    auto upload = [&](const Inputs& in) {
        query.upload(in.query, stream.value); sinks.upload(in.sinks, stream.value);
        table.upload(in.table, stream.value); lengths.upload(in.lengths, stream.value);
        starts.upload(in.starts, stream.value); active.upload(in.active, stream.value);
        check(cudaMemsetAsync(output.data, 0xff, output.count * sizeof(float), stream.value), "poison output");
        check(cudaMemsetAsync(workspace.data, 0xcd, workspace.count, stream.value), "poison workspace");
    };
    auto validate = [&](const Inputs& in, const std::vector<float>* sameInputOutput) {
        const auto result = output.read(stream.value);
        const auto scratch = workspace.read(stream.value);
        const GptOssXqaLayout layout(spec.batch, qHeads, spec.logical);
        std::set<int> sampled{0, spec.batch / 2, spec.batch - 1};
        for (int b = 0; b < std::min(spec.batch, 14); ++b) sampled.insert(b);
        for (int b = 0; b < spec.batch; ++b) {
            const int expectedMode = modeOracle(spec, in, physical, b);
            int mode; uint32_t length;
            std::memcpy(&mode, scratch.data() + layout.modes + std::size_t(b) * 4, 4);
            std::memcpy(&length, scratch.data() + layout.lengths + std::size_t(b) * 4, 4);
            require(mode == expectedMode, "complete row-mode oracle");
            require(length == (expectedMode == 1 ? uint32_t(in.lengths[b]) : 1U), "normalized positive length");
            ++counts.modes[expectedMode]; ++counts.rows;
            for (int p = 0; p < spec.logical; ++p) {
                const auto index = std::size_t(b) * spec.logical + p;
                int normalized; std::memcpy(&normalized, scratch.data() + layout.table + index * 4, 4);
                const int raw = in.table[index];
                require(normalized == (raw >= 0 && raw < physical ? raw : 0), "all page entries sanitized");
                ++counts.tables;
            }
            for (int h = 0; h < qHeads; ++h) {
                std::array<float, 64> expected{};
                if (sampled.count(b)) expected = attentionOracle(spec, in, physical, b, h, perLayer, keySeed, valueSeed);
                for (int d = 0; d < 64; ++d) {
                    const auto index = (std::size_t(b) * qHeads + h) * 64 + d;
                    const float actual = result[index];
                    require(std::isfinite(actual), "all output values finite");
                    if (!expectedMode) { require(actual == 0.f, "invalid/inactive row zero"); ++counts.zeros; }
                    if (sampled.count(b)) {
                        const double error = std::abs(double(actual) - double(expected[d]));
                        counts.maxAbs = std::max(counts.maxAbs, error); ++counts.checked;
                        if (error > .006 * (1 + std::abs(double(expected[d]))))
                            throw std::runtime_error("FP64 XQA bound failed case=" + std::to_string(caseIndex) +
                                " row=" + std::to_string(b) + " head=" + std::to_string(h) +
                                " dim=" + std::to_string(d) + " actual=" + std::to_string(actual) +
                                " expected=" + std::to_string(expected[d]));
                    }
                }
            }
        }
        require(keys.read(stream.value) == initialKeys && values.read(stream.value) == initialValues,
            "entire external KV cache remains bitwise unchanged");
        if (sameInputOutput) require(result == *sameInputOutput, "ordinary/captured same-input exact output");
        ++counts.invocations;
        return result;
    };
    const auto first = makeInputs(spec, physical, 0), second = makeInputs(spec, physical, 1);
    upload(first); invoke(); const auto firstOutput = validate(first, nullptr);
    upload(second); invoke(); const auto secondOutput = validate(second, nullptr);
    // No alloc/initialization/upload inside capture. Resources outlive graph;
    // changed query/table/length/active/sink data are supplied on its stream.
    {
        Graph captured(stream.value);
        check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal), "begin capture");
        invoke();
        check(cudaStreamEndCapture(stream.value, &captured.graph), "end capture");
        check(cudaGraphInstantiate(&captured.executable, captured.graph, nullptr, nullptr, 0), "graph instantiate");
        for (int revision = 0; revision < 2; ++revision) {
            const auto& in = revision ? second : first;
            upload(in);
            check(cudaGraphLaunch(captured.executable, stream.value), "changed graph replay");
            validate(in, revision ? &secondOutput : &firstOutput);
        }
    }
    ++counts.cases;
    std::cout << "CASE " << caseIndex << " B=" << spec.batch << " KVH=" << spec.kvHeads
        << " pages=" << spec.logical << " physical=" << physical << " window=" << spec.window
        << " layer=" << spec.layer << " pattern=" << spec.pattern << " ordinary/captured PASS\n" << std::flush;
}
} // namespace

int main() {
    try {
        int devices = 0; check(cudaGetDeviceCount(&devices), "device inventory");
        require(devices > 0, "GPU required; no compile-only run may pass native proof");
        const std::vector<Spec> specs{
            {1,16,0,1,0,1,false,1}, {1,16,23,4,1,2,false,17},
            {1,160,0,4,0,1,false,2005}, {1,160,128,4,1,3,false,2005},
            {1,256,0,16,1,4,false,4096}, {1,16,0,1,0,1,false,-1},
            {16,16,0,4,0,0}, {16,160,23,1,1,0}, {16,256,128,16,0,0},
            {128,160,0,4,0,0}, {128,160,128,4,1,0},
            {144,160,0,4,1,0}, {144,256,23,4,0,0},
            {448,16,0,4,0,1,true}, {448,160,128,4,1,0},
            {512,160,0,4,0,0}, {512,256,23,4,1,0}, {512,256,128,16,1,0}};
        Counts counts;
        const int tested = std::min(devices, 2);
        for (int device = 0; device < tested; ++device) {
            check(cudaSetDevice(device), "select execution device");
            std::cout << "DEVICE " << device << " sequential independent native tests\n" << std::flush;
            for (std::size_t i = 0; i < specs.size(); ++i) runCase(specs[i], int(i), counts);
        }
        require(counts.modes[0] && counts.modes[1] && counts.modes[2], "all dispatch modes exercised");
        std::cout << "XQA_NATIVE_PARITY_ACTUAL_PASS devices=" << tested << " cases=" << counts.cases
            << " invocations=" << counts.invocations << " rows=" << counts.rows << " table_entries=" << counts.tables
            << " sampled_fp64_values=" << counts.checked << " zero_values=" << counts.zeros
            << " guards=" << counts.guards << " max_abs=" << counts.maxAbs << " bound=.006*(1+abs)\n"
            << "SCOPE native sampled FP64/full safety metadata/read-only KV/ordinary+changed graph; "
            << "NOT compiled/plugin/pretrained/performance or all-four goal proof\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "XQA_NATIVE_PARITY_FAILED " << error.what() << '\n';
        return 1;
    }
}
