"""Retain the original independent FP64/control bounds and padding checks."""
from pathlib import Path
import sys
ns={'__file__':__file__}
wrapper=Path(__file__).with_name('make_decode_subsimd_probe.py').read_text()
exec(compile(wrapper[:wrapper.rindex('Path(sys.argv[1]).write_bytes')],__file__,'exec'),ns)
source=ns['source']
source=source.replace('{"bf16_decode_s16_t64",1,16,4,64,nil},{"bf16_decode_s16_t128",1,32,4,128,nil}', '{"bf16_decode_s8_t64",1,16,2,64,nil},{"bf16_decode_s8_t128",1,32,2,128,nil}')
needle='            auto weight=[device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];'
assert source.count(needle)==1
source=source.replace(needle,needle+'''
            tiles[0].name=columns<=256?"bf16_decode_small":"bf16_decode";
            tiles[0].columns=columns<=256?4:16;tiles[0].threads=columns<=256?64:128;
            tiles[0].pipeline=pipeline(device,library,tiles[0].name);
''')
assert 'bf16_decode_s16_' not in source and 'bf16_decode_s8_t64' in source
Path(sys.argv[1]).write_bytes(source.encode())
