"""Keep all original projections; retain native special keys in combined graphs."""
from pathlib import Path
import sys

source=Path(__file__).with_name('combined_graph_prefill_gpu.mm').read_text()
needle='size_t n,k;Garnet::MetalBF16Weight';assert source.count(needle)==1
source=source.replace(needle,'bool nativeKey=false;size_t n,k;Garnet::MetalBF16Weight')
needle='p.nativeWeight=native.UploadBF16';assert source.count(needle)==1
source=source.replace(needle,'p.nativeKey=key;p.nativeWeight=native.UploadBF16')
needle='size_t total=0;';assert source.count(needle)==1
source=source.replace(needle,'size_t total=0,nativeKeys=std::count_if(projections.begin(),projections.end(),[](const Projection& p){return p.nativeKey;});')
needle='            for(int index=0;index<2;++index){auto graph=';assert source.count(needle)==1
source=source.replace(needle,'            if(p.nativeKey)continue;\n'+needle)
needle='if(policy){int index=policy-1;';assert source.count(needle)==1
source=source.replace(needle,'if(policy){if(nativeKeys){native.Begin();for(auto& p:projections)if(p.nativeKey)native.EncodeBF16MatVec(p.input,p.nativeWeight,p.output);native.SubmitAndWait();}int index=policy-1;')
needle='for(auto& p:projections){auto actual=static_cast<const float*>(p.graphOutput[policy-1].contents);';assert source.count(needle)==1
source=source.replace(needle,'for(auto& p:projections){std::vector<float> retained;if(p.nativeKey){retained=native.Download(p.output);need(retained.size()==p.control.size()&&std::memcmp(retained.data(),p.control.data(),retained.size()*sizeof(float))==0,"Native key fallback changed original bits");}auto actual=p.nativeKey?retained.data():static_cast<const float*>(p.graphOutput[policy-1].contents);')
needle='@"projections":@(projections.size()),';assert source.count(needle)==1
source=source.replace(needle,needle+'@"native_key_fallbacks":@(nativeKeys),@"graph_projections":@(projections.size()-nativeKeys),@"native_key_bit_exact":@YES,')
Path(sys.argv[1]).write_bytes(source.encode())
