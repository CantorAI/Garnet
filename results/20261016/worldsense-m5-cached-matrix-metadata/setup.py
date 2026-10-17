from pathlib import Path
import shutil,json,hashlib,datetime,re,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-lazy-defaults/source';s=r/'out/research/worldsense-cached-matrix-metadata/source';assert not s.exists()
w.joinpath('design.json').write_text(json.dumps(dict(experiment='WS-M5-34',declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed='Frozen WS15 source; its rejection retained, no unchanged retry',hypothesis='Immutable native matrix storage/layout metadata avoids repeated MPS property calls and local autorelease pools in alias/layout/inventory checks while retaining WS15 exact lazy-default storage retirement. No attribution of prior slowdowns or assumed speed gain.',source_evidence='metal_context.mm constructs MetalMatrix::Impl and assigns its matrix at one Allocate site only. MetalMatrix copies share the same Impl; no external Objective-C mutation API. Numeric buffer/device identity is valid only while its Impl owns the MPSMatrix; no extra retained buffer or process-global pointer cache.',change='Default-OFF CACHED_MATRIX_METADATA requires LAZY_RESIDENT_DEFAULTS; cache numeric identity/device/rows/columns/rowBytes/offset/length/allocated bytes at allocation. SharesStorage, CompatibleLayout, IsDefaultLayout and MatrixBufferInfo use cached values. All other pools, host operations, encoding, model/runtime source and native arithmetic unchanged. C++ live-property audit test runs outside timing.',validation='OFF/ON native references + metadata/live property parity across empty/copy/move/reassignment/distinct same/different padded shapes, host update/zero, pending GPU output/Submit/Abort and surviving another context. Full32VLM/28text finite states/tokens/prefix/capacity/vision; ordinary/resident graph alternation, two sessions, close/rebuild/invalid controls and partial allocation faults.',protocol={'resource':'Fresh8realWorldSensecalls >=10% sampledMetal and >=5% sampled process footprint, exact/golden/native/identities.','speed':'Only ifresourcepasses,NEW20realWorldSensecalls bothmodelsTTFT/total/decode andnativeSkillwall <=2%regression.','confirmation':'Onlyifspeedpasses,NEW48calls mirrored4workers/twoindependentlypassingfull3frame/querypairs beforepromotion.'},tradeoffs='Extra numeric metadata per matrix; allocation-time properties and checks counted in initialization. Pointer identities are not physical residency proof. Does not optimize GPU arithmetic; host savings may be immaterial. All old rejected runs retained. No source/build/GPU/archive jobs during performance. Preserve CUDA, BF16/FP32, context/image budgets and exact output gates.'),indent=2))
shutil.copytree(old,s,ignore=shutil.ignore_patterns('__pycache__','.git','results'))
p=s/'cmake/GarnetMetal.cmake';text=p.read_text();text+='''
option(GARNET_METAL_CACHED_MATRIX_METADATA "Immutable native matrix storage/layout metadata" OFF)
if(GARNET_METAL_CACHED_MATRIX_METADATA)
 if(NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS)
  message(FATAL_ERROR "Cached matrix metadata cohort requires lazy default storage")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_CACHED_MATRIX_METADATA=1)
endif()
add_executable(garnet_matrix_metadata_test test2026/metal/matrix_metadata_test.cpp)
target_link_libraries(garnet_matrix_metadata_test PRIVATE garnet_metal)
add_test(NAME metal_matrix_metadata_reference COMMAND garnet_matrix_metadata_test)
''';p.write_text(text)
p=s/'src/backends/metal/metal_context.mm';text=p.read_text();needle='    MPSMatrix* matrix;';assert text.count(needle)==1;text=text.replace(needle,needle+'''
#ifdef GARNET_METAL_CACHED_MATRIX_METADATA
    // The MPSMatrix and its storage layout are assigned only in Allocate.
    // Numeric identities never outlive this owning Impl; no extra buffer owner.
    uintptr_t bufferIdentity=0,deviceIdentity=0;
    size_t rows=0,columns=0,rowBytes=0,offset=0;
    uint64_t allocatedBytes=0,lengthBytes=0;
#endif''')
for sig,nextsig,body in [
 ('bool MetalMatrix::SharesStorage(', 'bool MetalMatrix::CompatibleLayout(','''    return impl_&&other.impl_&&impl_->bufferIdentity==other.impl_->bufferIdentity;'''),
 ('bool MetalMatrix::CompatibleLayout(', 'struct MetalBF16Weight::Impl', '''    return impl_&&other.impl_&&impl_->deviceIdentity==other.impl_->deviceIdentity&&
        impl_->rows==other.impl_->rows&&impl_->columns==other.impl_->columns&&impl_->rowBytes==other.impl_->rowBytes;'''),
 ('bool MetalContext::IsDefaultLayout(', 'MetalBufferInfo MetalContext::MatrixBufferInfo(','''    if(!value.impl_)return false;const auto& m=*value.impl_;
    const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
    return m.deviceIdentity==reinterpret_cast<uintptr_t>((__bridge void*)impl_->device)&&m.rows==rows&&m.columns==columns&&
        m.rowBytes==stride&&m.offset==0&&m.lengthBytes==rows*stride;'''),
 ('MetalBufferInfo MetalContext::MatrixBufferInfo(', 'std::vector<MetalBufferInfo> MetalContext::WeightBufferInfo(','''    if(!value.impl_)return {};const auto& m=*value.impl_;
    return {m.bufferIdentity,m.allocatedBytes,m.lengthBytes,"matrix_fp32"};''')]:
 start=text.index(sig);end=text.index(nextsig,start);oldbody=text[start:end];brace=oldbody.index('{');assert oldbody.rstrip().endswith('}')
 text=text[:start]+oldbody[:brace+1]+'\n#ifdef GARNET_METAL_CACHED_MATRIX_METADATA\n'+body+'\n#else\n'+oldbody[brace+1:oldbody.rfind('}')]+ '\n#endif\n}\n'+text[end:]
needle='    storage->matrix=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];';assert text.count(needle)==1;text=text.replace(needle,needle+'''
#ifdef GARNET_METAL_CACHED_MATRIX_METADATA
    if(!storage->matrix)throw std::runtime_error("Metal matrix view allocation failed");
    auto matrix=storage->matrix;
    storage->bufferIdentity=reinterpret_cast<uintptr_t>((__bridge void*)matrix.data);
    storage->deviceIdentity=reinterpret_cast<uintptr_t>((__bridge void*)matrix.device);
    storage->rows=matrix.rows;storage->columns=matrix.columns;storage->rowBytes=matrix.rowBytes;storage->offset=matrix.offset;
    storage->allocatedBytes=matrix.data.allocatedSize;storage->lengthBytes=matrix.data.length;
#endif''')
text+='''
namespace Garnet {
bool MetalContext::MatrixMetadataMatchesLive(const MetalMatrix& value) {
    @autoreleasepool {
        if(!value.impl_)return true;
#ifdef GARNET_METAL_CACHED_MATRIX_METADATA
        const auto& m=*value.impl_;auto native=m.matrix;id<MTLBuffer> buffer=native.data;
        return native&&buffer&&m.bufferIdentity==reinterpret_cast<uintptr_t>((__bridge void*)buffer)&&
            m.deviceIdentity==reinterpret_cast<uintptr_t>((__bridge void*)native.device)&&m.rows==native.rows&&m.columns==native.columns&&
            m.rowBytes==native.rowBytes&&m.offset==native.offset&&m.allocatedBytes==buffer.allocatedSize&&m.lengthBytes==buffer.length;
#else
        return value.impl_->matrix&&value.impl_->matrix.data;
#endif
    }
}
}
''';p.write_text(text)
p=s/'src/backends/metal/metal_context.h';text=p.read_text();needle='    static MetalBufferInfo MatrixBufferInfo(';assert needle in text;text=text.replace(needle,'    // Correctness diagnostic only; deliberately queries live MPS properties.\n    static bool MatrixMetadataMatchesLive(const MetalMatrix&);\n'+needle);p.write_text(text)
p=s/'src/entry/metal_main.cpp';text=p.read_text();needle='#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS';i=text.index(needle);text=text[:i]+'''#ifdef GARNET_METAL_CACHED_MATRIX_METADATA
    result += " cached_matrix_metadata";
#endif
'''+text[i:];# determine backend string variable from original before writing
if 'result +=' not in p.read_text():
 match=re.search(r'(\w+)\s*\+=\s*"[^"]*lazy_resident_defaults',p.read_text());assert match; text=text.replace('result += " cached_matrix_metadata"',match.group(1)+' += " cached_matrix_metadata"')
p.write_text(text)
(w/'seed-source-manifest.json').write_text(json.dumps({str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(old.rglob('*'))if p.is_file() and '__pycache__'not in p.parts},indent=2))
print('WS34 design and source prepared',s)
