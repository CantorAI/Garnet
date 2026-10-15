# SPDX-License-Identifier: Apache-2.0
"""Fixed-grid Qwen still-image encoder through native Garnet Metal/MPS."""
import garnet
from .vision_encoder import CPUImageEncoder

class MetalImageEncoder(CPUImageEncoder):
    def __init__(self,checkpoint,config,grid_h,grid_w,batch=1):
        if not garnet.backend_info().startswith('Garnet Metal/MPS '):
            raise ValueError('MetalImageEncoder requires the native Garnet Metal package')
        CPUImageEncoder.__init__(self,checkpoint,config,grid_h,grid_w,batch)
