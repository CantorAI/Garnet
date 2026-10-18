#include <metal_stdlib>
using namespace metal;
kernel void raw_bf16_expand_words(device const ushort* weight [[buffer(0)]],device uint* output [[buffer(1)]],
 constant ulong& start [[buffer(2)]],constant ulong& count [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(ulong(id)<count)output[id]=uint(weight[start+ulong(id)])<<16;
}
