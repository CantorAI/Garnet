from pathlib import Path
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-primary-u32'
t=(seed/'primary_corpus_test.cpp').read_text()
a=' std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;uint64_t wordsChecked=0,packedMatrices=0,rawMatrices=0,rawBytes=0,primaryBytes=0;';assert t.count(a)==1
t=t.replace(a,a+'\n std::map<std::string,uint64_t> allTileWords{{"32x32",0},{"64x32",0},{"64x64",0}};',1)
a='  uint64_t checked=0;';assert t.count(a)==1;t=t.replace(a,a+'\n  std::map<std::string,uint64_t> tileWords{{"32x32",0},{"64x32",0},{"64x64",0}};',1)
a='   ++packedMatrices;wordsChecked+=checked;';assert t.count(a)==1
t=t.replace(a,'''   for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,32},{64,64}}) {
    const auto key=std::to_string(geometry.first)+"x"+std::to_string(geometry.second);
    const size_t rowChunk=(8*1024*1024/inner/geometry.first)*geometry.first;need(rowChunk>0,"Learned tile row chunk is empty");
    for(size_t first=0;first<rows;first+=rowChunk) {
     const size_t count=std::min(rowChunk,rows-first);auto actual=gpu.PrimaryReadPrefillTileWords(w,first,count,geometry.first,geometry.second);
     need(actual.size()==count*inner,"Learned prefill tile readback size mismatch");
     for(size_t i=0;i<actual.size();++i){size_t at=(first*inner+i)*2;uint32_t original=uint32_t(data[at])|(uint32_t(data[at+1])<<8);need(actual[i]==(original<<16),"Actual checkpoint prefill tile FP32 word mismatch");}
     tileWords[key]+=actual.size();allTileWords[key]+=actual.size();
    }
    need(tileWords[key]==checked,"Learned prefill tile word coverage mismatch");
   }
'''+a,1)
a='{"GPU_vector_original_FP32_words_checked",checked},';assert t.count(a)==1;t=t.replace(a,a+'{"GPU_prefill_tile_FP32_words_checked",tileWords},',1)
a='{"GPU_vector_original_FP32_words_checked",wordsChecked}};';assert t.count(a)==1;t=t.replace(a,'{"GPU_vector_original_FP32_words_checked",wordsChecked},{"GPU_prefill_tile_FP32_words_checked",allTileWords}};',1)
a=' report["status"]="full_actual_both_readers_packed_corpus_GPU_bits_exact_not_inference_validated";';assert t.count(a)==1
t=t.replace(a,' for(const auto& entry:allTileWords)need(entry.second==5921308672ULL,"Complete actual prefill tile words missing");\n report["status"]="full_actual_two_original_and_three_prefill_tile_readers_GPU_bits_exact_not_inference_validated";',1)
(w/'primary_corpus_test.cpp').write_text(t)
print('All398 learned-weight corpus extended to all3 new shared-loader tile readers; original2 readers unchanged.')
