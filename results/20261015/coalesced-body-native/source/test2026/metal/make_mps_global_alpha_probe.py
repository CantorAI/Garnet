"""Toy API gate: put the exact global inverse power-of-two into MPS alpha."""
from pathlib import Path
import sys
source=Path(__file__).with_name('probe_mps_exact_half.mm').read_text()
needle='interiorColumns:k alpha:1 beta:0';assert source.count(needle)==1
source=source.replace(needle,'interiorColumns:k alpha:std::ldexp(1.0,-16) beta:0')
source=source.replace('actual=std::ldexp(actual,-16);','')
source=source.replace('error>0.0002*(1+std::abs(expected))','error>5e-6+5e-5*std::abs(expected)')
source=source.replace('mps_exact_half_probe','mps_global_alpha_probe')
source=source.replace('@"weight_scale_exponent":@16','@"weight_scale_exponent":@16,@"alpha":@(std::ldexp(1.0,-16)),@"separate_rescale_dispatch":@NO')
Path(sys.argv[1]).write_bytes(source.encode())
