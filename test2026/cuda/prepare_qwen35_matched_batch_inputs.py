"""Unique decoded real-image variants for every round/group/batch-row."""
import hashlib,json,sys
from pathlib import Path
from PIL import Image
from qwen35_benchmark_workload import IMAGES,batch_inputs
source,target=map(Path,sys.argv[1:3])
assert not target.exists(),'Preserve prior datasets; choose a new destination'
records=[];decoded_hashes=set()
for trial in range(7):
    for group in range(4):
        names,paths=batch_inputs(str(target),trial,group,8,True)
        for row,(name,path) in enumerate(zip(names,paths)):
            src=source/name
            with Image.open(src) as image:image=image.convert('RGB').resize((320,192),Image.Resampling.BILINEAR)
            salt=trial*32+group*8+row
            image.paste(((salt*37)%256,(salt*73+19)%256,(salt*109+41)%256),(312,184,320,192))
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path,quality=95)
            with Image.open(path) as decoded:decoded_hash=hashlib.sha256(decoded.convert('RGB').tobytes()).hexdigest()
            assert decoded_hash not in decoded_hashes,'Decoded variants collided'
            decoded_hashes.add(decoded_hash)
            records.append({'trial':trial,'group':group,'row':row,'image':name,'relative_path':str(path.relative_to(target)),
                'jpeg_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoded_rgb_sha256':decoded_hash,
                'source':str(src.resolve()),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(target/'manifest.json').write_text(json.dumps({'transform':'RGB bilinear320x192, distinct8x8 corner tile per invocation, JPEG95',
    'scope':'Same files for all engines; batch1/2/4 use first rows of each group; not unmodified originals',
    'records':records},indent=2))
print('MATCHED_BATCH_INPUTS_COMPLETE',len(records),'unique decoded variants')
