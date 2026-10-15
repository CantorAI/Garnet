# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 still-image tower, fixed grid with one image per batch row."""
from .tensor_compat import tensor
import garnet
import tensor as native_tensor
T=tensor()
@T.fusion(name='qwen35_vision_encoder',role='vision_encoder',boundary='required')
def Qwen35VisionEncoder(patches,config,grid_h,grid_w):
    return patches * T.unary_op('qwen35_vision_encoder',grid_h=grid_h,grid_w=grid_w,
        hidden_size=config['hidden_size'],num_heads=config['num_heads'],depth=config['depth'],
        patch_size=config['patch_size'],temporal_patch_size=config['temporal_patch_size'],in_channels=config['in_channels'],
        spatial_merge_size=config['spatial_merge_size'],num_position_embeddings=config['num_position_embeddings'],
        out_hidden_size=config['out_hidden_size'],rope_theta=config.get('rope_parameters',{}).get('rope_theta',10000))

class CPUImageEncoder:
    def __init__(self,checkpoint,config,grid_h,grid_w,batch=1):
        if batch not in (1,2):raise ValueError('fixed batch must be 1 or 2')
        self.shape=[batch,grid_h*grid_w,config['in_channels']*config['temporal_patch_size']*config['patch_size']**2]
        symbol=native_tensor.input('still_image_patches',shape=self.shape,dtype=native_tensor.float32)
        @T.fusion(name='qwen35_cpu_still_image',boundary='required')
        def forward(patches):return Qwen35VisionEncoder(patches,config,grid_h,grid_w)
        self.graph=garnet.compiled_graph()
        try:self.graph.build(forward(symbol),[symbol],checkpoint)
        except:self.graph.close();raise
        self.io=garnet.tensor_io()
    def encode_file(self,path):return self.graph.run([self.io.read_f32(path,self.shape)])
    def close(self):self.graph.close()
