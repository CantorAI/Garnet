"""CPU immutable Git-source proof; no native/GPU/model/throughput claim."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'tools/gpt_oss'))
from audit_resident_session import ENGINE_SOURCE_PATHS,profile_source_identity

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);git=root/'repo';git.mkdir()
    def command(*args):return subprocess.check_output(['git','-C',str(git),*args],text=True).strip()
    command('init','-q');command('config','user.name','CPU source proof');command('config','user.email','cpu-proof@example.invalid')
    names=[]
    for selected in ENGINE_SOURCE_PATHS:
        name=selected+'/fixture.py' if '.' not in Path(selected).name else selected
        p=git/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU immutable engine-program fixture\n');names.append(name)
    command('add','.');command('commit','-qm','CPU original source');measured=command('rev-parse','HEAD')
    (git/'docs.txt').write_text('CPU unrelated reporting change\n');command('add','.');command('commit','-qm','CPU reporting only')
    observed=command('rev-parse','HEAD');profile=dict(source_commit=measured)
    snapshot=root/'snapshot'
    for name in names:
        p=snapshot/'Garnet'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((git/name).read_bytes())
    proof=profile_source_identity(profile,observed,git,snapshot)
    assert proof['measured_source_commit']==measured and proof['execution_source_commit']==observed
    assert len(proof['engine_program_blobs'])==8 and proof['archived_execution_bytes_verified']
    assert len(proof['engine_program_manifest_sha256'])==64
    rejected=0
    def reject(profile,execution,repository=None,archive=None):
        global rejected
        try:profile_source_identity(profile,execution,repository,archive)
        except (ValueError,subprocess.CalledProcessError):rejected+=1
        else:raise AssertionError('Invalid historical source compatibility accepted')
    reject(profile,observed) # A mismatched origin is not silently allowed.
    reject(dict(source_commit='bad'),observed,git)
    victim=snapshot/'Garnet'/names[0];original=victim.read_bytes();victim.write_bytes(original+b'corruption')
    reject(profile,observed,git,snapshot);victim.write_bytes(original)
    victim.unlink();reject(profile,observed,git,snapshot);victim.write_bytes(original)
    # Every declared source group, including ownership/KV/admission/model/core,
    # changes the proof even though no production math is executed in this test.
    for index,name in enumerate(names):
        p=git/name;original=p.read_bytes();p.write_bytes(original+b'changed engine program\n')
        command('add','.');command('commit','-qm','CPU changed engine source')
        reject(profile,command('rev-parse','HEAD'),git)
        p.write_bytes(original);command('add','.');command('commit','-qm','CPU restore fixture bytes')
    # A missing required closure group cannot become an empty green manifest.
    victim=git/names[-1];victim.unlink();command('add','-A');command('commit','-qm','CPU missing engine helper')
    reject(profile,command('rev-parse','HEAD'),git)
    same=profile_source_identity(profile,measured)
    assert same['compatibility']=='same immutable commit'
print('CPU immutable profile-origin proof: reporting-only reuse, exact eight source groups/archived bytes and',rejected,
    'changed/missing source/proof hazards rejected; no native/GPU proof')
