"""Replace only direct memset with original memcpy from a retained zero vector."""
from pathlib import Path
import sys
source=Path(__file__).with_name('joint_state_zero_gpu.mm').read_text()
needle='  auto condition=[&]()';assert source.count(needle)==1
source=source.replace(needle,'''  size_t maximum=0;for(const auto& m:matrices)if(m.rows*m.columns>maximum)maximum=m.rows*m.columns;
  std::vector<float> cachedZero(maximum,0.0f);
'''+needle)
needle='else std::memset(target,0,m.columns*4);';assert source.count(needle)==1
source=source.replace(needle,'else std::memcpy(target,cachedZero.data()+row*m.columns,m.columns*4);')
source=source.replace('joint_state_zero_failed','reused_zero_source_failed')
Path(sys.argv[1]).write_bytes(source.encode())
