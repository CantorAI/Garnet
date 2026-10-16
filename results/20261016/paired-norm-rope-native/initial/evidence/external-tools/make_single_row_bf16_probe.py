"""Single-row specialization of original BF16 math; batch two uses control."""
from pathlib import Path
import sys
source=Path(__file__).with_name('make_constant_bf16_probe.py').read_text()
source=source.replace('constant_bf16_decode','single_bf16_decode')
exec(compile(source,str(Path(__file__).with_name('make_constant_bf16_probe.py')),'exec'))
path=Path(sys.argv[1]);text=path.read_text()
needle='mode==2?recordable[small]:normal[small]';assert text.count(needle)==1
text=text.replace(needle,'mode==2&&rows==1?recordable[small]:normal[small]')
text=text.replace('@"weight_access":@"device-readonly versus constant"','@"row_specialization":@"single row; batch two retains original kernel"')
path.write_bytes(text.encode())
