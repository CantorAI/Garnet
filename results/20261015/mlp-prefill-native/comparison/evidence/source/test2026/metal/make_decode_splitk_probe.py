"""One command buffer for partial and reduction kernels; original strict bounds."""
from pathlib import Path
import sys
ns={'__file__':__file__}
wrapper=Path(__file__).with_name('make_decode_eight_lane_probe.py').read_text()
exec(compile(wrapper[:wrapper.rindex('Path(sys.argv[1]).write_bytes')],__file__,'exec'),ns)
source=ns['source'];start=source.index('        std::vector<Tile> tiles=');end=source.index('        for(auto& tile:',start)
source=source[:start]+'''        std::vector<Tile> tiles={{"bf16_decode",1,16,4,128,nil},{"bf16_decode_sk2",1,16,2,128,nil},{"bf16_decode_sk4",1,16,4,128,nil}};
        auto reduce2=pipeline(device,library,"bf16_decode_sk2_reduce"),reduce4=pipeline(device,library,"bf16_decode_sk4_reduce");
        if(reduce2.maxTotalThreadsPerThreadgroup<64||reduce4.maxTotalThreadsPerThreadgroup<64)throw std::runtime_error("Reduction exceeds device limits");
'''+source[end:]
needle='auto out=[device newBufferWithLength:rows*oStride*4 options:MTLResourceStorageModeShared];'
assert source.count(needle)==1
source=source.replace(needle,needle+'auto partials=[device newBufferWithLength:4*rows*oStride*4 options:MTLResourceStorageModeShared];')
needle='auto run=[&](const Tile& tile) {'
assert source.count(needle)==1
source=source.replace(needle,needle+'bool split=std::strstr(tile.name,"bf16_decode_sk")!=nullptr;')
source=source.replace('[enc setBuffer:out offset:0 atIndex:2]','[enc setBuffer:(split?partials:out) offset:0 atIndex:2]')
source=source.replace('MTLSizeMake((columns+tile.columns-1)/tile.columns,1,1)','MTLSizeMake(((columns+tile.columns-1)/tile.columns)*(split?tile.inner:1),1,1)')
needle='[enc endEncoding];finish(cb);'
assert source.count(needle)==1
source=source.replace(needle,'''[enc endEncoding];
                    if(split){auto reduce=[cb computeCommandEncoder];[reduce setComputePipelineState:tile.inner==2?reduce2:reduce4];
                        [reduce setBuffer:partials offset:0 atIndex:0];[reduce setBuffer:out offset:0 atIndex:1];[reduce setBytes:&p length:sizeof(p) atIndex:2];
                        [reduce dispatchThreadgroups:MTLSizeMake((rows*columns+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[reduce endEncoding];}
                    finish(cb);''')
assert 'bf16_decode_s8_' not in source
Path(sys.argv[1]).write_bytes(source.encode())
