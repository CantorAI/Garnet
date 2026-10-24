"""Required retained full-model evidence before a joined-MLP HTTP pilot."""
def validate_joined_model_gates(original,joined,full,golden,library_sha256):
    for report,mode,nv in ((original,'original',192),(joined,'joined',128)):
        if report.get('status')!='passed' or report.get('mode')!=mode or report.get('library_sha256')!=library_sha256:
            raise ValueError('Wrong joined state qualification status/mode/binary')
        program=report['program'];graph=program['graph'];pool=report['pool']
        if graph['native_nvfp4_projection_weights']!=nv or graph['native_fp8_projection_weights']!=208 or graph['external_bf16_weight_inputs']!=0 or graph['graph_capture_failures']!=0 or pool['compressed_projection_pool_entries']!=nv+208:
            raise ValueError('Wrong state qualification native counts')
        width=program['hidden_width']+program['state_width']+program['payload_width']
        if width!=174587904 or len(report['cases'])!=6:
            raise ValueError('Incomplete full T128/cap4096 packed coverage')
        if {(row['case'],row['repeat']) for row in report['cases']}!={(case,repeat) for case in ('text','image') for repeat in range(3)}:
            raise ValueError('Missing state input/repeat')
        for row in report['cases']:
            offset=0
            for chunk in row['chunks']:
                if chunk['offset']!=offset or type(chunk['words']) is not int or not 0<chunk['words']<=1048576:
                    raise ValueError('State chunk gap/overlap or invalid width')
                digest=chunk['sha256']
                if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid state digest')
                offset+=chunk['words']
            if offset!=width:raise ValueError('Incomplete state hash coverage')
    if joined.get('same_profile_full_output_exact') is not True or original['inputs']!=joined['inputs'] or original['cases']!=joined['cases']:
        raise ValueError('Original/joined full packed output mismatch')
    physical=lambda report:report['pool']['compressed_projection_weight_bytes']+report['pool']['compressed_projection_scale_bytes']
    if physical(original)!=physical(joined):raise ValueError('Joined pool changed physical projection bytes')
    if full.get('status')!='smoke_passed' or full.get('library_sha256')!=library_sha256:
        raise ValueError('Missing same-binary full generation gate')
    for key,value in [('batch',1),('capacity',4096),('prefill_chunk',128),('decode_chunk',4)]:
        if type(full.get(key)) is not int or full[key]!=value:raise ValueError('Wrong full generation profile')
    if len(full['cases'])!=12 or len(golden['cases'])!=12:raise ValueError('Incomplete full generation cases')
    expected={(row['key'],row['repeat']):row for row in golden['cases']}
    if len(expected)!=12 or {(row['key'],row['repeat']) for row in full['cases']}!=set(expected):raise ValueError('Missing generation repeat')
    for row in full['cases']:
        prior=expected[(row['key'],row['repeat'])]
        if any(row[field]!=prior[field] for field in ('request_sha256','prompt_ids','generated_ids','normalized_sha256')):
            raise ValueError('Joined generation differs from retained original')
    memory=full['decoder_final'];profiles=memory['packed_profiles']
    if profiles.get('joined_mlp_projections') is not True or memory['compressed_projection_pool_entries']!=336:
        raise ValueError('Full generation did not use joined pool')
    graphs=memory['graphs']
    if len(graphs)!=3 or any(g['graph_capture_failures']!=0 for g in graphs):raise ValueError('Unexpected full graphs/capture failure')
    for graph in graphs[:2]:
        if graph['native_nvfp4_projection_weights']!=128 or graph['native_fp8_projection_weights']!=208 or graph['external_compressed_weight_inputs']!=336 or graph['external_bf16_weight_inputs']!=0:
            raise ValueError('Full generation projection count/fallback mismatch')
    return {'full_state_cases':6,'full_generation_cases':12,'shared_projection_matrices':336,
            'scope':'Retained T128 state hash coverage and short-input T128/T4 generation gates; no nonzero-state T4 byte, MTP, broad-quality, long-context or performance claim.'}
