"""Match existing reference_dense FP32 stage boundaries; retain original bounds."""
from pathlib import Path
import sys

source=Path(__file__).with_name('dependent_mlp_graph_gpu.mm').read_text()
old='std::vector<double> activated(p.intermediate);';assert source.count(old)==1
source=source.replace(old,'std::vector<float> activated(p.intermediate);')
old='activated[j]=(g/(1.+std::exp(-g)))*u;';assert source.count(old)==1
source=source.replace(old,'float gate=static_cast<float>(g),up=static_cast<float>(u);activated[j]=(gate/(1.f+std::exp(-gate)))*up;')
old='p.reference[r*p.width+c]=sum;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(sum);';assert source.count(old)==1
source=source.replace(old,'float rounded=static_cast<float>(sum);p.reference[r*p.width+c]=rounded;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(double(rounded));')
Path(sys.argv[1]).write_bytes(source.encode())
