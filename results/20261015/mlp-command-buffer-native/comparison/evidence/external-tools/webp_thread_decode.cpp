// CPU-only exact-input experiment; use the same libwebp as Pillow.
#include <webp/decode.h>
#include <cstddef>
#include <cstdint>
#include <limits>
extern "C" int garnet_webp_version(){return WebPGetDecoderVersion();}
extern "C" int garnet_webp_info(const uint8_t* data,size_t bytes,int* width,int* height) {
    WebPBitstreamFeatures features;
    if(!data||!width||!height||WebPGetFeatures(data,bytes,&features)!=VP8_STATUS_OK)return -1;
    if(features.has_alpha||features.has_animation||features.width<=0||features.height<=0)return -2;
    *width=features.width;*height=features.height;return 0;
}
extern "C" int garnet_webp_decode(const uint8_t* data,size_t bytes,int threads,uint8_t* output,size_t capacity) {
    WebPDecoderConfig config;
    if(!output||!WebPInitDecoderConfig(&config)||WebPGetFeatures(data,bytes,&config.input)!=VP8_STATUS_OK)return -1;
    if(config.input.has_alpha||config.input.has_animation||config.input.width<=0||config.input.height<=0)return -2;
    size_t width=size_t(config.input.width),height=size_t(config.input.height);
    if(width>size_t(std::numeric_limits<int>::max()/3)||height>std::numeric_limits<size_t>::max()/(width*3)||capacity<width*height*3)return -3;
    config.options.use_threads=threads?1:0;
    config.output.colorspace=MODE_RGB;config.output.is_external_memory=1;
    config.output.u.RGBA.rgba=output;config.output.u.RGBA.stride=int(width*3);config.output.u.RGBA.size=capacity;
    auto status=WebPDecode(data,bytes,&config);WebPFreeDecBuffer(&config.output);return int(status);
}
