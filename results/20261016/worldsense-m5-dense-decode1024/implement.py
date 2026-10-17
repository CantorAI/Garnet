from pathlib import Path
import shutil
old=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-decode512/source');new=old.parent.parent/'worldsense-dense-decode1024/source';assert not new.exists();shutil.copytree(old,new)
def edit(rel,fn):
 p=new/rel;p.write_text(fn(p.read_text()))
p=new/'src/backends/metal/dense_decode1024_shader.h';p.write_text((old/'src/backends/metal/dense_decode512_shader.h').read_text().replace('DenseDecode512','DenseDecode1024').replace('DENSE512','DENSE1024').replace('dense512','dense1024').replace('ulong(group)*64+simd*4','ulong(group)*128+simd*4'))
def context(s):
 s=s.replace('#include "dense_decode512_shader.h"\n#endif','#include "dense_decode512_shader.h"\n#endif\n#ifdef GARNET_METAL_DENSE_DECODE1024\n#include "dense_decode1024_shader.h"\n#endif')
 s=s.replace('denseDecode512Encoded=0;','denseDecode512Encoded=0,denseDecode1024Encoded=0;')
 s=s.replace('source=[source stringByAppendingString:[NSString stringWithUTF8String:kDenseDecode512Shader]];\n#endif','source=[source stringByAppendingString:[NSString stringWithUTF8String:kDenseDecode512Shader]];\n#endif\n#ifdef GARNET_METAL_DENSE_DECODE1024\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kDenseDecode1024Shader]];\n#endif')
 s=s.replace('#ifdef GARNET_METAL_DENSE_DECODE512\n        constexpr','#ifdef GARNET_METAL_DENSE_DECODE1024\n        constexpr size_t wideColumns=128,wideThreads=1024;NSString* wideName=@"specialized_bf16_decode_dense1024";\n#elif defined(GARNET_METAL_DENSE_DECODE512)\n        constexpr')
 s=s.replace('#if defined(GARNET_METAL_DENSE_DECODE256) || defined(GARNET_METAL_DENSE_DECODE512)','#if defined(GARNET_METAL_DENSE_DECODE256) || defined(GARNET_METAL_DENSE_DECODE512) || defined(GARNET_METAL_DENSE_DECODE1024)')
 s=s.replace('#ifdef GARNET_METAL_DENSE_DECODE512\n                if(found->second.wide)', '#ifdef GARNET_METAL_DENSE_DECODE1024\n                if(found->second.wide){threads=1024;decodeColumns=128;denseWideSelected=true;}\n#elif defined(GARNET_METAL_DENSE_DECODE512)\n                if(found->second.wide)')
 needle='''#ifdef GARNET_METAL_DENSE_DECODE512
        if(denseWideSelected&&threads==512&&decodeColumns==64&&a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256&&o.columns<100000)++impl_->denseDecode512Encoded;
#endif''';assert needle in s;s=s.replace(needle,needle+'\n'+needle.replace('512','1024').replace('decodeColumns==64','decodeColumns==128'))
 needle='size_t MetalContext::DenseDecode512EncodedTotal() const {return impl_->denseDecode512Encoded;}';assert needle in s;s=s.replace(needle,needle+'\n'+needle.replace('512','1024'));return s
edit('src/backends/metal/metal_context.mm',context)
for rel in ['src/backends/metal/metal_context.h','src/backends/metal/decoder_session.h']:
 edit(rel,lambda s:s.replace('    size_t DenseDecode512EncodedTotal() const;','    size_t DenseDecode512EncodedTotal() const;\n    size_t DenseDecode1024EncodedTotal() const;'))
edit('src/backends/metal/decoder_session.h',lambda s:s.replace('        APISET().AddProp("dense_decode512_encoded_total",&MetalDecoderSession::DenseDecode512EncodedTotal);','        APISET().AddProp("dense_decode512_encoded_total",&MetalDecoderSession::DenseDecode512EncodedTotal);\n        APISET().AddProp("dense_decode1024_encoded_total",&MetalDecoderSession::DenseDecode1024EncodedTotal);'))
edit('src/backends/metal/decoder_session.cpp',lambda s:s+'\nnamespace Garnet {size_t MetalDecoderSession::DenseDecode1024EncodedTotal() const {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->DenseDecode1024EncodedTotal():0;}}\n')
edit('runtime/metal_serving.py',lambda s:s.replace("        stages['dense_decode512_encoded_total'] = self.decoder.session.dense_decode512_encoded_total","        stages['dense_decode512_encoded_total'] = self.decoder.session.dense_decode512_encoded_total\n        stages['dense_decode1024_encoded_total'] = self.decoder.session.dense_decode1024_encoded_total"))
edit('src/entry/metal_main.cpp',lambda s:s.replace('#ifdef GARNET_METAL_DENSE_PREFIX_CACHE','#ifdef GARNET_METAL_DENSE_DECODE1024\n        info+="; dense_decode1024";\n#endif\n#ifdef GARNET_METAL_DENSE_PREFIX_CACHE',1))
(new/'test2026/metal/dense_decode1024_test.cpp').write_text((old/'test2026/metal/dense_decode512_test.cpp').read_text().replace('DENSE_DECODE512','DENSE_DECODE1024').replace('DenseDecode512','DenseDecode1024').replace('Dense512','Dense1024'))
edit('cmake/GarnetMetal.cmake',lambda s:s+'''\noption(GARNET_METAL_DENSE_DECODE1024 "Thirty-two SIMD groups for inner2048/6144 body decode" OFF)
if(GARNET_METAL_DENSE_DECODE1024)
 if(NOT GARNET_METAL_BODY_SHAPE OR GARNET_METAL_DENSE_DECODE256 OR GARNET_METAL_DENSE_DECODE512)
  message(FATAL_ERROR "Dense decode1024 requires body shape and excludes decode256/512")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DENSE_DECODE1024=1)
endif()
add_executable(garnet_dense_decode1024_test test2026/metal/dense_decode1024_test.cpp)
target_include_directories(garnet_dense_decode1024_test PRIVATE src/backends/metal)
target_link_libraries(garnet_dense_decode1024_test PRIVATE garnet_metal)
add_test(NAME metal_dense_decode1024_reference COMMAND garnet_dense_decode1024_test "${CMAKE_CURRENT_BINARY_DIR}/dense_decode1024_words.bin")
''')
w=Path(__file__).resolve().parent
s=(w.parent/'worldsense-dense-decode512/build.py').read_text().replace('worldsense-dense-decode512','worldsense-dense-decode1024').replace("'-DGARNET_METAL_DENSE_DECODE512='+mode.upper(),","'-DGARNET_METAL_DENSE_DECODE512=OFF','-DGARNET_METAL_DENSE_DECODE1024='+mode.upper(),");(w/'build.py').write_text(s)
s=(w.parent/'worldsense-dense-decode512/primitive_checks.py').read_text().replace('dense-decode512','dense-decode1024').replace('dense_decode512','dense_decode1024').replace('native11','native12');(w/'primitive_checks.py').write_text(s)
print('Isolated1024source implemented;not built or tested;old256/512OFF bothbuilds')
