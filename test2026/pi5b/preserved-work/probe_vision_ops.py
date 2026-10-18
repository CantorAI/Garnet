import openvino as ov,numpy as np,math
op=ov.opset13
x=np.linspace(-10,10,20001,dtype=np.float32)
p=op.parameter(x.shape,np.float32)
for approximation in ['TANH','ERF']:
    model=ov.Model([op.gelu(p,approximation)],[p])
    ref=.5*x.astype(np.float64)*(1+(np.tanh(np.sqrt(2/np.pi)*(x.astype(np.float64)+.044715*x.astype(np.float64)**3)) if approximation=='TANH' else np.vectorize(math.erf)(x.astype(np.float64)/np.sqrt(2))))
    for mode in ['PERFORMANCE','ACCURACY']:
        compiled=ov.Core().compile_model(model,'CPU',{'INFERENCE_PRECISION_HINT':'f32','EXECUTION_MODE_HINT':mode,'INFERENCE_NUM_THREADS':2})
        actual=compiled([x])[0]
        print(approximation,mode,'max_abs',np.max(np.abs(actual-ref)),flush=True)
