"""Shared explicit paths for historical and cache-busting batch cohorts."""
import os,hashlib
IMAGES=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg']
def batch_inputs(dataset,trial,group,batch,matched=False):
    names=[IMAGES[(group+b)%4] for b in range(batch)]
    paths=[os.path.join(dataset,str(trial),str(group),str(b),name) if matched else os.path.join(dataset,str(trial),name) for b,name in enumerate(names)]
    return names,paths
def input_hashes(paths):
    result=[]
    for path in paths:
        with open(path,'rb') as stream:result.append(hashlib.sha256(stream.read()).hexdigest())
    return result
