from pathlib import Path
s=Path('/Users/shawn/CantorAI/out/research/worldsense-vector-prefill/source');p=s/'test2026/metal/primary_storage_test.cpp';t=p.read_text()
helper='''
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
static void checkPrefillTiles(MetalContext& gpu,const MetalBF16Weight& weight,const std::vector<uint16_t>& bits,size_t rows,size_t inner) {
 const auto before=MetalContext::PrimaryStats();
 for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,32},{64,64}}) {
  auto actual=gpu.PrimaryReadPrefillTileWords(weight,0,rows,geometry.first,geometry.second);
  need(actual.size()==bits.size(),"Prefill tile full word count mismatch");
  for(size_t i=0;i<bits.size();++i)need(actual[i]==uint32_t(bits[i])<<16,"Prefill tile changed original FP32 bits");
  const size_t first=((rows-1)/geometry.first)*geometry.first,count=rows-first;
  auto tail=gpu.PrimaryReadPrefillTileWords(weight,first,count,geometry.first,geometry.second);
  need(tail.size()==count*inner,"Prefill tile tail word count mismatch");
  for(size_t i=0;i<tail.size();++i)need(tail[i]==uint32_t(bits[first*inner+i])<<16,"Prefill tile nonzero-row/tail bits mismatch");
 }
 const auto after=MetalContext::PrimaryStats();need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Prefill diagnostic retained a primary owner");
}
#endif
'''
a='int main(){try{';assert t.count(a)==1;t=t.replace(a,helper+'\n'+a,1)
a=' w=MetalBF16Weight();need(MetalContext::PrimaryStats().activeWeights==now.activeWeights,"Shared owner was freed");';assert t.count(a)==1;t=t.replace(a,'#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL\n checkPrefillTiles(gpu,w,bits,512,128);\n#endif\n'+a,1)
a='need(actual[i]==uint32_t(mix[i])<<16,"Mixed raw/compact GPU bits mismatch");';assert t.count(a)==1
t=t.replace(a,a+'''
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
 checkPrefillTiles(gpu,w,mix,260,256);
 for(int which=0;which<6;++which) {
  bool bad=false;
  try{gpu.PrimaryReadPrefillTileWords(w,which==0?1:which==2?260:0,which==1?0:which==3?4:64,which==4?16:64,which==5?16:32);}catch(...){bad=true;}
  need(bad,"Prefill tile malformed range/geometry accepted");
 }
 gpu.Begin();{bool bad=false;try{gpu.PrimaryReadPrefillTileWords(w,0,64,64,32);}catch(...){bad=true;}need(bad,"Prefill tile admitted an open batch");}gpu.Abort();
 need(gpu.PrimaryReadPrefillTileWords(w,256,4,64,32)[0]==uint32_t(mix[256*256])<<16,"Prefill tile recovery after Abort failed");
#endif''',1)
a=' std::vector<uint16_t> raw(257*129,0x3f80);';assert t.count(a)==1
t=t.replace(a,'''
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
 {
  // Valid shape/range with >8Miwords isolates the readback cap from bounds.
  std::vector<uint16_t> large(65540*128,0x3f80);auto weight=gpu.UploadBF16(large.data(),large.size()*2,65540,128);
  bool bad=false;try{gpu.PrimaryReadPrefillTileWords(weight,0,65540,64,32);}catch(...){bad=true;}need(bad,"Prefill tile readback cap not enforced");
 }
 std::cout<<"Vector prefill tile all3 geometries/full finite/mixed words/nonzero rows/tail260/closed-batch/cap/recovery PASS"<<std::endl;
#endif
'''+a,1)
p.write_text(t)
print('Meaningful full finite/mixed/tail/negative/bounded-reader native cases added; original186projectionrecords untouched.')
