"""Check whether the native CLI can prepare the canonical image patches.

No learned operators execute here. Arguments: source, model, metadata, output.
The outer CPython probe verifies the resulting bytes against the saved fixture.
"""
import sys,json
sys.path.insert(0,sys.argv[1]+'/test2026/pi5b')
import numpy as np
from PIL import Image
from prepare_cpu_image import prepare
folder=sys.argv[2]
with open(folder+'/config.json') as stream:config=json.load(stream)['vision_config']
with open(folder+'/preprocessor_config.json') as stream:processor=json.load(stream)
with open(sys.argv[3]) as stream:metadata=json.load(stream)
with Image.open(metadata['image']) as image:
    patches,grid,source,resized=prepare(image,config,processor,65536,65536)
values=np.tile(patches,(metadata['shape'][0],1,1)).astype('<f4')
values.tofile(sys.argv[4])
print('xlang_image_preparation_complete',flush=True)
