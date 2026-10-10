#include "q4q8_linear.h"
#include <cmath>
#include <iostream>
#include <vector>
int main() {
 for(int group : {32,64,128}) {
  Garnet::QuantizedWeight w; w.shape={9,128}; w.blockSize=group;
  w.packedValues.resize(9*128/2); w.fp16Scales.assign(9*128/group,0x3c00);
  std::vector<float> x(128),expected(9),out(9),cached(9);
  for(int c=0;c<128;++c) x[c]=(c%group==0)?127:((c*19)%255-127);
  for(int r=0;r<9;++r) for(int c=0;c<128;++c) {
   int value=(r*7+c*3)%16-8; size_t i=r*128+c;
   w.packedValues[i/2]|=(value&15)<<((i%2)*4);
   expected[r]+=value*x[c];
  }
  Garnet::Q4Q8Linear kernel; std::string error;
  if(!kernel.Prepare(w,error)){std::cerr<<error;return 1;}
  Garnet::Q4Q8Linear view;
  if(!view.BindPackedView(9,128,group,kernel.PackedWeightData(),kernel.PackedWeightBytes(),kernel.PackedScaleData(),kernel.PackedScaleCount(),error)) return 2;
  for(int threads : {1,2,4}) {
   if(!kernel.Forward(x.data(),out.data(),threads,error)||!view.Forward(x.data(),cached.data(),threads,error)) {std::cerr<<error;return 3;}
   for(int r=0;r<9;++r) if(std::abs(out[r]-expected[r])>0.01f||out[r]!=cached[r]) {std::cerr<<"numerical mismatch";return 4;}
  }
  std::cout<<"group="<<group<<" threads=1,2,4 canonical/reference/packed-view PASS\n";
 }
}
