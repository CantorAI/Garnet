from pathlib import Path
import shutil
old=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-decode256/source');new=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-decode512/source');assert not new.exists();shutil.copytree(old,new)
p=new/'src/backends/metal/dense_decode512_shader.h';s=(old/'src/backends/metal/dense_decode256_shader.h').read_text().replace('kDenseDecode256Shader','kDenseDecode512Shader').replace('DENSE256','DENSE512').replace('specialized_bf16_decode_dense256','specialized_bf16_decode_dense512').replace('ulong(group)*32+simd*4','ulong(group)*64+simd*4');p.write_text(s)
p=new/'src/backends/metal/metal_context.mm';s=p.read_text();needle='#include "dense_decode256_shader.h"\n#endif';assert needle in s;s=s.replace(needle,needle+'\n#ifdef GARNET_METAL_DENSE_DECODE512\n#include "dense_decode512_shader.h"\n#endif');s=s.replace('    size_t denseDecode256Encoded=0;','    size_t denseDecode256Encoded=0,denseDecode512Encoded=0;');needle='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kDenseDecode256Shader]];\n#endif';assert needle in s;s=s.replace(needle,needle+'\n#ifdef GARNET_METAL_DENSE_DECODE512\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kDenseDecode512Shader]];\n#endif')
needle='        auto create=[&](bool wide)->id<MTLComputePipelineState> {';assert needle in s;s=s.replace(needle,'''#ifdef GARNET_METAL_DENSE_DECODE512
        constexpr size_t wideColumns=64,wideThreads=512;NSString* wideName=@"specialized_bf16_decode_dense512";
#else
        constexpr size_t wideColumns=32,wideThreads=256;NSString* wideName=@"specialized_bf16_decode_dense256";
#endif
'''+needle);s=s.replace('wide?32:(small?4:16)','wide?wideColumns:(small?4:16)').replace('wide?@"specialized_bf16_decode_dense256":','wide?wideName:').replace('wide?256:(small?64:128)','wide?wideThreads:(small?64:128)')
s=s.replace('#ifdef GARNET_METAL_DENSE_DECODE256\n        if(a.rows==1', '#if defined(GARNET_METAL_DENSE_DECODE256) || defined(GARNET_METAL_DENSE_DECODE512)\n        if(a.rows==1')
s=s.replace('#ifdef GARNET_METAL_DENSE_DECODE256\n        bool dense256Selected=false;', '#if defined(GARNET_METAL_DENSE_DECODE256) || defined(GARNET_METAL_DENSE_DECODE512)\n        bool denseWideSelected=false;')
needle='#ifdef GARNET_METAL_DENSE_DECODE256\n                if(found->second.wide){threads=256;decodeColumns=32;dense256Selected=true;}\n#endif';assert needle in s;s=s.replace(needle,'''#ifdef GARNET_METAL_DENSE_DECODE512
                if(found->second.wide){threads=512;decodeColumns=64;denseWideSelected=true;}
#elif defined(GARNET_METAL_DENSE_DECODE256)
                if(found->second.wide){threads=256;decodeColumns=32;denseWideSelected=true;}
#endif''');s=s.replace('if(dense256Selected&&','if(denseWideSelected&&');needle='''        if(denseWideSelected&&threads==256&&decodeColumns==32&&a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256&&o.columns<100000)++impl_->denseDecode256Encoded;
#endif''';assert needle in s;s=s.replace(needle,needle+'''\n#ifdef GARNET_METAL_DENSE_DECODE512
        if(denseWideSelected&&threads==512&&decodeColumns==64&&a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256&&o.columns<100000)++impl_->denseDecode512Encoded;
#endif''');needle='size_t MetalContext::DenseDecode256EncodedTotal() const {return impl_->denseDecode256Encoded;}';assert needle in s;s=s.replace(needle,needle+'\nsize_t MetalContext::DenseDecode512EncodedTotal() const {return impl_->denseDecode512Encoded;}');p.write_text(s)
p=new/'src/backends/metal/metal_context.h';s=p.read_text().replace('    size_t DenseDecode256EncodedTotal() const;','    size_t DenseDecode256EncodedTotal() const;\n    size_t DenseDecode512EncodedTotal() const;');p.write_text(s)
p=new/'src/backends/metal/decoder_session.h';s=p.read_text().replace('    size_t DenseDecode256EncodedTotal() const;','    size_t DenseDecode256EncodedTotal() const;\n    size_t DenseDecode512EncodedTotal() const;');needle='        APISET().AddProp("dense_decode256_encoded_total",&MetalDecoderSession::DenseDecode256EncodedTotal);';assert needle in s;s=s.replace(needle,needle+'\n        APISET().AddProp("dense_decode512_encoded_total",&MetalDecoderSession::DenseDecode512EncodedTotal);');p.write_text(s)
p=new/'src/backends/metal/decoder_session.cpp';p.write_text(p.read_text()+'\nnamespace Garnet {size_t MetalDecoderSession::DenseDecode512EncodedTotal() const {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->DenseDecode512EncodedTotal():0;}}\n')
p=new/'runtime/metal_serving.py';s=p.read_text();needle="        stages['dense_decode256_encoded_total'] = self.decoder.session.dense_decode256_encoded_total";assert needle in s;s=s.replace(needle,needle+"\n        stages['dense_decode512_encoded_total'] = self.decoder.session.dense_decode512_encoded_total");p.write_text(s)
p=new/'src/entry/metal_main.cpp';s=p.read_text();needle='#ifdef GARNET_METAL_DENSE_PREFIX_CACHE';at=s.index(needle);s=s[:at]+'#ifdef GARNET_METAL_DENSE_DECODE512\n        info+="; dense_decode512";\n#endif\n'+s[at:];p.write_text(s)
p=new/'test2026/metal/dense_decode512_test.cpp';p.write_text((old/'test2026/metal/dense_decode256_test.cpp').read_text().replace('DENSE_DECODE256','DENSE_DECODE512').replace('DenseDecode256EncodedTotal','DenseDecode512EncodedTotal').replace('Dense256','Dense512'))
p=new/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+'''\noption(GARNET_METAL_DENSE_DECODE512 "Sixteen SIMD groups for inner2048/6144 body decode" OFF)
if(GARNET_METAL_DENSE_DECODE512)
 if(NOT GARNET_METAL_BODY_SHAPE OR GARNET_METAL_DENSE_DECODE256)
  message(FATAL_ERROR "Dense decode512 requires body shape and excludes decode256")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DENSE_DECODE512=1)
endif()
add_executable(garnet_dense_decode512_test test2026/metal/dense_decode512_test.cpp)
target_include_directories(garnet_dense_decode512_test PRIVATE src/backends/metal)
target_link_libraries(garnet_dense_decode512_test PRIVATE garnet_metal)
add_test(NAME metal_dense_decode512_reference COMMAND garnet_dense_decode512_test "${CMAKE_CURRENT_BINARY_DIR}/dense_decode512_words.bin")
''')
w=Path('work/worldsense-dense-decode512');s=Path('work/worldsense-dense-decode256/build.py').read_text().replace('worldsense-dense-decode256','worldsense-dense-decode512');needle="'-DGARNET_METAL_DENSE_DECODE256='+mode.upper(),";assert needle in s;s=s.replace(needle,"'-DGARNET_METAL_DENSE_DECODE256=OFF','-DGARNET_METAL_DENSE_DECODE512='+mode.upper(),");(w/'build.py').write_text(s)
print('Isolated512 prototype written;unbuilt/untested;256OFF inbothbuilds')
