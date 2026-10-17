from pathlib import Path
import subprocess,difflib,json,hashlib
r=Path('/Users/shawn/CantorAI');w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-wide-cached-prefill');s=r/'out/research/worldsense-wide-cached-prefill/source';old=r/'out/research/worldsense-wide-primary-lazy/source';base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
subprocess.run(['git','apply','--check','--exclude=src/entry/metal_main.cpp','--exclude=test2026/metal/primary_storage_test.cpp',str(w/'wide46-over-f21.patch')],cwd=s,check=True);subprocess.run(['git','apply','--exclude=src/entry/metal_main.cpp','--exclude=test2026/metal/primary_storage_test.cpp',str(w/'wide46-over-f21.patch')],cwd=s,check=True)
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='        return info;';assert t.count(needle)==1;p.write_text(t.replace(needle,'#ifdef GARNET_METAL_WIDE_PRIMARY_STORAGE\n        info+="; wide_primary_storage";\n#endif\n'+needle,1))
p=s/'test2026/metal/primary_storage_test.cpp';t=p.read_text();a=subprocess.check_output(['git','-C',str(r/'Garnet'),'show',base+':test2026/metal/primary_storage_test.cpp']).decode().splitlines(keepends=True);b=(old/'test2026/metal/primary_storage_test.cpp').read_text().splitlines(keepends=True)
for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
 if op=='equal':continue
 assert op=='insert',op;addition=''.join(b[k:l]);anchor=a[i-1];assert t.count(anchor)==1,(anchor,t.count(anchor));t=t.replace(anchor,anchor+addition,1)
p.write_text(t)
# Read-only original raw storage diagnostic, never exposed to inference API.
p=s/'src/backends/metal/metal_context.h';t=p.read_text();a='    std::vector<uint32_t> PrimaryReadWords(const MetalBF16Weight&,size_t start,size_t count);';assert t.count(a)==1;p.write_text(t.replace(a,a+'\n    std::vector<uint32_t> RawBF16ReadWords(const MetalBF16Weight&,size_t start,size_t count);',1))
p=s/'src/backends/metal/metal_context.mm';t=p.read_text();t+='''
namespace Garnet {
std::vector<uint32_t> MetalContext::RawBF16ReadWords(const MetalBF16Weight& weight,size_t start,size_t count) {
 @autoreleasepool {
  if(impl_->command||impl_->submitted.count)throw std::logic_error("Raw weight readback requires closed drained batch");
  auto owner=weight.impl_;
  if(!owner||weight.PrimaryReady()||!owner->buffer||owner->buffer.device!=impl_->device||owner->buffer.storageMode!=MTLStorageModeShared)
   throw std::invalid_argument("Raw weight readback requires same-device shared original BF16 storage");
  const size_t total=Elements(owner->rows,owner->columns);
  if(!count||count>8*1024*1024||start>total||count>total-start||owner->buffer.length!=total*sizeof(uint16_t))
   throw std::invalid_argument("Raw weight readback range/length invalid");
  auto data=static_cast<const char*>(owner->buffer.contents);if(!data)throw std::runtime_error("Raw weight storage unavailable");
  std::vector<uint32_t> output(count);
  for(size_t i=0;i<count;++i){uint16_t word;std::memcpy(&word,data+2*(start+i),2);output[i]=uint32_t(word)<<16;}
  return output;
 }
}
}
''';p.write_text(t)
p=s/'test2026/metal/primary_storage_test.cpp';t=p.read_text();a=' MetalContext gpu;auto initial=MetalContext::PrimaryStats();';assert t.count(a)==1
check='''
 {
  const size_t rows=512,columns=128,count=rows*columns;std::vector<uint16_t> bits;
  for(uint32_t word=0;word<65536;++word)if((word&0x7f80)!=0x7f80)bits.push_back(uint16_t(word));bits.resize(count,0x8000);
  std::vector<char> unaligned(count*2+1);std::memcpy(unaligned.data()+1,bits.data(),count*2);
  auto raw=gpu.UploadBF16(unaligned.data()+1,count*2,rows,columns,false,false,false),alias=raw;
  std::fill(unaligned.begin(),unaligned.end(),0);raw=MetalBF16Weight();
  auto actual=gpu.RawBF16ReadWords(alias,0,count);need(actual.size()==count,"Raw word count changed");
  for(size_t i=0;i<count;++i)need(actual[i]==uint32_t(bits[i])<<16,"Raw finite BF16 word/source/alias changed");
  auto part=gpu.RawBF16ReadWords(alias,7,103);for(size_t i=0;i<part.size();++i)need(part[i]==uint32_t(bits[7+i])<<16,"Raw offset read changed");
  auto reject=[&](const MetalBF16Weight& x,size_t first,size_t n){bool bad=false;try{gpu.RawBF16ReadWords(x,first,n);}catch(...){bad=true;}need(bad,"Invalid raw readback admitted");};
  reject(MetalBF16Weight(),0,1);reject(alias,0,0);reject(alias,0,8*1024*1024+1);reject(alias,SIZE_MAX,1);reject(alias,count,1);
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
  auto packed=gpu.UploadBF16(bits.data(),count*2,rows,columns);need(packed.PrimaryReady(),"Raw negative test requires actual packed weight");reject(packed,0,1);
#endif
  gpu.Begin();reject(alias,0,1);gpu.Abort();need(gpu.RawBF16ReadWords(alias,0,1)[0]==uint32_t(bits[0])<<16,"Raw read after Abort changed");
  std::cout<<"Raw shared BF16 finite-pattern/unaligned/source-alias/range/cap/invalid/openbatch/Abort original-bit PASS"<<std::endl;
 }
 need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights,"Raw reader retained primary owners");
''';p.write_text(t.replace(a,a+check,1))
seed=json.loads((w/'ws52-seed-source-manifest.json').read_text());m={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()};assert len(m)==1717
protected=['src/backends/metal/primary_block_shader.h','src/backends/metal/primary_vector_prefill_shader.h','src/backends/metal/primary_block_codec.h','src/backends/metal/body_shape_shader.h','runtime/metal_serving.py','test2026/metal/primary_projection_test.cpp']
assert all(m[k]==seed[k]for k in protected);assert all(m[k]==h for k,h in seed.items()if '/cuda/'in k)
for name in ['starting-source-manifest.json','correctness-source-manifest.json']:(w/name).write_text(json.dumps(m,indent=2)+'\n')
basefiles=json.loads((Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-cached-lazy/seed-source-manifest.json')).read_text());changes=[k for k,h in m.items()if basefiles.get(k)!=h]
(w/'implementation-audit.json').write_text(json.dumps({'source_files':len(m),'changed_files':changes,'seed_files_changed':[k for k,h in m.items()if seed.get(k)!=h],'protected_shader_codec_wrapper_projection_CUDA_unchanged':True,'merge':'Sixwide46files applied directly; entry marker andtestinsertions manuallymerged fromsameimmutable46delta. Initial check failed onlyoverlapping52additions; no partialapply/native/build/performance occurred before reviewedmerge.','scope':'Wideclassifier/admission plus boundedHOSTrawstorage reader; no53numeric/performance evidence yet.'},indent=2)+'\n');print('53sourcefrozen',len(m),'canonicaloverlay',len(changes),'protectedinputs unchanged')
