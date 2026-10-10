"""Exact small RGB fixture checks channel/temporal/merge patch ordering."""
import numpy as np
from PIL import Image
from prepare_cpu_image import prepare
pixels=np.arange(48,dtype=np.uint8).reshape(4,4,3)
config=dict(patch_size=2,spatial_merge_size=2,temporal_patch_size=2)
processor=dict(image_mean=[.5,.5,.5],image_std=[.5,.5,.5])
patches,grid,source,resized=prepare(Image.fromarray(pixels),config,processor,16,16)
expected=[]
for h,w in [(0,0),(0,1),(1,0),(1,1)]:
    row=[]
    for c in range(3):
        for t in range(2):
            for ph in range(2):
                for pw in range(2):
                    value=np.float32(pixels[h*2+ph,w*2+pw,c])/np.float32(255)
                    row.append((value-np.float32(.5))/np.float32(.5))
    expected.append(row)
assert grid==(2,2) and source==(4,4) and resized==(4,4)
np.testing.assert_array_equal(patches,np.asarray(expected,dtype=np.float32))
print('CPU RGB normalization/temporal duplication/merge patch layout PASS')
