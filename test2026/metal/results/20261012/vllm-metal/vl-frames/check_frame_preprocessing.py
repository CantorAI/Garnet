import sys,numpy as np
from PIL import Image
sys.path.insert(0,'test2026/metal');sys.path.insert(0,'test2026/pi5b')
from frame_preprocessing import prepare_frame
from prepare_cpu_image import prepare
cfg=dict(patch_size=2,spatial_merge_size=2,temporal_patch_size=2);proc=dict(image_mean=[.5]*3,image_std=[.5]*3)
a=np.arange(48,dtype=np.uint8).reshape(4,4,3);image=Image.fromarray(a)
expected,_,_,_=prepare(image,cfg,proc,16,16)
for batch in (1,2):
 actual,rgb,hw=prepare_frame(image,cfg,proc,batch,16)
 np.testing.assert_array_equal(actual,np.tile(expected,(batch,1,1)))
 assert hw==(4,4) and rgb==a.tobytes()
print('Canonical frame preprocessing PASS exact RGB/temporal/channel/merge ordering batch 1/2')
