import numpy as np
import openvino as ov
from openvino import opset13 as op
x=np.linspace(-4,4,257,dtype=np.float32)
p=op.parameter(x.shape,ov.Type.f32)
soft=op.add(op.maximum(p,op.constant(np.float32(0))),op.log(op.add(op.constant(np.float32(1)),op.exp(op.negative(op.absolute(p))))))
m=ov.Model([op.softplus(p),op.exp(p),op.sigmoid(p),soft],[p])
for mode in ('PERFORMANCE','ACCURACY'):
    c=ov.Core().compile_model(m,'CPU',{'INFERENCE_PRECISION_HINT':'f32','EXECUTION_MODE_HINT':mode})
    y=c([x])
    for i,ref in enumerate((np.logaddexp(0,x),np.exp(x),1/(1+np.exp(-x)),np.logaddexp(0,x))):
        a=y[c.output(i)]
        print(mode,i,'max absolute',np.max(np.abs(a-ref)),'max relative',np.max(np.abs(a-ref)/np.maximum(np.abs(ref),1e-10)))
