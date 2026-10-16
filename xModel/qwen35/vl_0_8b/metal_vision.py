# SPDX-License-Identifier: Apache-2.0
"""Fixed-grid Qwen still-image encoder through native Garnet Metal/MPS."""
import garnet
from .vision_encoder import CPUImageEncoder

class MetalImageEncoder(CPUImageEncoder):
    def __init__(self,checkpoint,config,grid_h,grid_w,batch=1,shared_encoder=None):
        if not garnet.backend_info().startswith('Garnet Metal/MPS '):
            raise ValueError('MetalImageEncoder requires the native Garnet Metal package')
        self._weight_source = shared_encoder.graph if shared_encoder is not None else None
        try:CPUImageEncoder.__init__(self,checkpoint,config,grid_h,grid_w,batch)
        finally:self._weight_source = None

    def _build_graph(self,output,symbols,checkpoint):
        if self._weight_source is None:
            self.graph.build(output,symbols,checkpoint)
        else:
            self.graph.build_shared(output,symbols,checkpoint,self._weight_source)
