"""Read actual identifier-attributed drafting evidence; requested flags are not proof."""
import hashlib,json,re
from pathlib import Path

def lmstudio_speculation(log,identifier):
    path=Path(log)
    if not path.exists():return {'status':'unknown','reason':'Runtime log unavailable'}
    content=path.read_bytes();contexts=0;drafts=[]
    for line_number,line in enumerate(content.decode(errors='replace').splitlines(),1):
        try:record=json.loads(line)
        except json.JSONDecodeError:continue
        data=record.get('data',{})
        if data.get('modelIdentifier')!=identifier:continue
        text=data.get('message','')
        contexts+=text.count('creating MTP draft context')
        for accepted,generated in re.findall(r'draft acceptance\s*=.*?\(\s*(\d+) accepted\s*/\s*(\d+) generated\)',text):
            drafts.append({'line':line_number,'accepted':int(accepted),'generated':int(generated)})
    generated=sum(d['generated'] for d in drafts)
    return {'status':'enabled_observed' if generated else 'unknown',
        'method':'MTP' if contexts else 'unresolved',
        'runtime_log':str(path),'runtime_log_sha256':hashlib.sha256(content).hexdigest(),
        'mtp_context_creation_events':contexts,'draft_timing_events':len(drafts),
        'accepted_draft_tokens':sum(d['accepted'] for d in drafts),'generated_draft_tokens':generated,
        'draft_records':drafts,'configured_depth_verified':False,
        'limitation':'Active drafts are runtime evidence. Missing events do not prove speculation disabled; exact proposal settings are unresolved.'}

def saved_comparator_speculation(root,size,batch,lm_suffix='-partitioned',vllm_suffix=''):
    root=Path(root);stem=f'matched512-lmstudio-{size}-b{batch}{lm_suffix}'
    config_path=root/(stem+'.config.json')
    lm_config=json.loads(config_path.read_text()) if config_path.exists() else {}
    identifier=lm_config.get('request_config',{}).get('model',f'matched512-{size}-b{batch}{lm_suffix}')
    log=root/(stem+'-runtime.jsonl')
    if not log.exists():log=root/(identifier.replace(f'-{size}',f'-lmstudio-{size}',1)+'-runtime.jsonl')
    lm=lmstudio_speculation(log,identifier)
    path=root/f'matched512-vllm-{size}-b{batch}{vllm_suffix}.config.json'
    config=json.loads(path.read_text()) if path.exists() else {}
    spec=config.get('options',{}).get('speculative_config')
    return {'lmstudio':lm,'vllm':{'status':'enabled_requested' if spec else 'disabled_requested' if config else 'unknown',
        'speculative_config':spec,'evidence':str(path)},'configuration_equivalent':False}
