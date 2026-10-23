"""Common real-image JPEG fixtures with per-round cache-busting corner pixels.

All engines receive byte-identical files. The small resolution bounds vision
work at 60 merged tokens; original-source hashes and transforms are retained.
"""
import sys,json,hashlib
from pathlib import Path
from PIL import Image
source,target=map(Path,sys.argv[1:3]);target.mkdir(parents=True,exist_ok=True)
names=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg'];records=[]
for name in names:
    path=source/name
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:base=image.convert('RGB').resize((320,192),Image.Resampling.BILINEAR)
    for trial in range(7):
        folder=target/str(trial);folder.mkdir(exist_ok=True)
        image=base.copy();image.putpixel((319,191),(trial*31+1,17,29));image.save(folder/name,quality=95)
        records.append({'trial':trial,'image':name,'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest(),
            'source_sha256':source_hash,'source':str(path.resolve()),'width':320,'height':192,'visual_tokens':60})
manifest={'transform':'PIL RGB bilinear resize320x192; one corner pixel changed per round; JPEG quality95',
    'reason':'Same bounded image work for all engines; force image re-evaluation where prefix caching cannot be disabled.',
    'prefix_cache_limitation':'An engine can still reuse a short text-only prefix before the fresh image.',
    'records':records}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('QWEN35_BENCHMARK_INPUTS_COMPLETE',len(records))
