"""Adapt the established matrix harness with independent FP64 checks."""
from pathlib import Path
import sys
source=Path(__file__).with_name('decode_wide_outputs_gpu.mm').read_text()
start=source.index('        std::vector<Tile> tiles=');end=source.index('        for(auto& tile:',start)
source=source[:start]+'''        std::vector<Tile> tiles={{"bf16_decode",1,16,4,128,nil},{"bf16_decode_s16_t64",1,16,4,64,nil},{"bf16_decode_s16_t128",1,32,4,128,nil}};
'''+source[end:]
needle='                for(size_t variant=1;variant<tiles.size();++variant) {'
assert source.count(needle)==1
source=source.replace(needle,'''                std::vector<double> exact(rows*columns),bounds(rows*columns);
                auto raw=static_cast<const uint16_t*>(bytes.bytes);
                const double epsilon=std::ldexp(1.0,-23),n=std::ceil(double(inner)/32)+32;
                const double gamma=n*epsilon/(1-n*epsilon);
                double controlRatio=0;
                for(uint64_t row=0;row<rows;++row)for(uint64_t col=0;col<columns;++col) {
                    double sum=0,absolute=0;
                    for(uint64_t k=0;k<inner;++k) {
                        uint32_t bits=uint32_t(raw[col*inner+k])<<16;float w;std::memcpy(&w,&bits,4);
                        double product=double(w)*double(input[row*aStride+k]);sum+=product;absolute+=std::abs(product);
                    }
                    auto index=row*columns+col;exact[index]=sum;bounds[index]=gamma*absolute+1e-7;
                    if(!std::isfinite(sum)||!std::isfinite(absolute)||!std::isfinite(reference[row*oStride+col]))
                        throw std::runtime_error("Nonfinite independent reference/control");
                    controlRatio=std::max(controlRatio,std::abs(double(reference[row*oStride+col])-sum)/bounds[index]);
                }
                if(controlRatio>1)throw std::runtime_error("Control exceeds independent FP64 envelope");
                for(size_t variant=1;variant<tiles.size();++variant) {''')
source=source.replace('auto actual=static_cast<float*>(out.contents);double squaredError=0,squaredValue=0;',
    'auto actual=static_cast<float*>(out.contents);double fp64Ratio=0,comparisonRatio=0,squaredError=0,squaredValue=0;')
needle='double error=double(actual[index])-reference[index];'
assert source.count(needle)==1
source=source.replace(needle,'''double error=double(actual[index])-reference[index];
                        fp64Ratio=std::max(fp64Ratio,std::abs(double(actual[index])-exact[row*columns+col])/bounds[row*columns+col]);
                        comparisonRatio=std::max(comparisonRatio,std::abs(error)/(5e-6+5e-5*std::abs(double(reference[index]))));''')
needle='if(std::memcmp(actual,reference.data(),reference.size()*4)!=0)throw std::runtime_error("Work-distribution variant changed output bits");'
assert source.count(needle)==1
source=source.replace(needle,'if(fp64Ratio>1||comparisonRatio>1)throw std::runtime_error("Sub-SIMD exceeds unchanged numerical bounds");')
needle='std::printf("{\\"kind\\":\\"check\\",'
start=source.index(needle);end=source.index('std::fflush(stdout);',start)
source=source[:start]+'''std::printf("{\\"kind\\":\\"check\\",\\"matrix\\":\\"%s\\",\\"rows\\":%llu,\\"variant\\":\\"%s\\",\\"bits_exact\\":%s,\\"max_abs_error\\":%.9g,\\"fp64_worst_ratio\\":%.9g,\\"control_fp64_worst_ratio\\":%.9g,\\"comparison_worst_ratio\\":%.9g,\\"padding_preserved\\":true}\\n",name.UTF8String,(unsigned long long)rows,tile.name,exact?"true":"false",maxError,fp64Ratio,controlRatio,comparisonRatio);'''+source[end:]
# The existing harness uses 'exact' for the bit-comparison boolean. Rename the
# independent reference vector so both declarations remain unambiguous.
source=source.replace('std::vector<double> exact(','std::vector<double> fp64(').replace('exact[index]=sum','fp64[index]=sum').replace('-exact[row*columns+col]','-fp64[row*columns+col]')
Path(sys.argv[1]).write_bytes(source.encode())
