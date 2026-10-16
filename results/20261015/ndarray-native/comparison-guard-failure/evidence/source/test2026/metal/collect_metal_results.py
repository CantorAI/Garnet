import json,tarfile
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');names=['modelscope-download-20261011','pretrained-binary-20261012','pretrained-vision-saturated-20261012','vision-row-norm-20261012','text-b1-20261012','text-long-b1-20261012','text-long-b2-20261012','image-b1-20261012','image-b2-20261012','image-row-norm-b1-20261012','image-row-norm-b2-20261012','image-long-b1-20261012']
with tarfile.open(root/'logs/validated-results-20261012.tar.gz','w:gz') as archive:
 for name in names:
  folder=root/'logs'/name;exit=json.loads((folder/'exit.json').read_text())
  if exit['exit_code']!=0:raise ValueError(name+' not successful')
  for file in ('exit.json','output.log'):archive.add(folder/file,arcname=name+'/'+file)
print('Collected',len(names),'completed jobs')
