"""CPU metadata fixtures; these never constitute actual GPU qualification."""
import copy,unittest
from test2026.cuda.qwen38_5090.joined_model_gates import validate_joined_model_gates

def fixtures():
    chunks=[];width=174587904
    for offset in range(0,width,1048576):chunks.append({'offset':offset,'words':min(1048576,width-offset),'sha256':'1'*64})
    original={'status':'passed','mode':'original','library_sha256':'binary','inputs':{'text':{'sha256':'2'*64},'image':{'sha256':'3'*64}},
      'program':{'hidden_width':655360,'state_width':173932544,'payload_width':0,'graph':{'native_nvfp4_projection_weights':192,'native_fp8_projection_weights':208,'external_bf16_weight_inputs':0,'graph_capture_failures':0}},
      'pool':{'compressed_projection_pool_entries':400,'compressed_projection_weight_bytes':1024,'compressed_projection_scale_bytes':128},
      'cases':[{'case':case,'repeat':repeat,'chunks':copy.deepcopy(chunks)} for case in ('text','image') for repeat in range(3)]}
    joined=copy.deepcopy(original);joined.update(mode='joined',same_profile_full_output_exact=True)
    joined['program']['graph']['native_nvfp4_projection_weights']=128;joined['pool']['compressed_projection_pool_entries']=336
    cases=[{'key':key,'repeat':repeat,'request_sha256':'request','prompt_ids':[1,2],'generated_ids':[10,99],'normalized_sha256':'normalized'} for key in ('text','image','red','green','blue','bottom-blue') for repeat in range(2)]
    graph={'native_nvfp4_projection_weights':128,'native_fp8_projection_weights':208,'external_compressed_weight_inputs':336,'external_bf16_weight_inputs':0,'graph_capture_failures':0}
    full={'status':'smoke_passed','batch':1,'capacity':4096,'prefill_chunk':128,'decode_chunk':4,'library_sha256':'binary','cases':copy.deepcopy(cases),
      'decoder_final':{'packed_profiles':{'joined_mlp_projections':True},'compressed_projection_pool_entries':336,'graphs':[copy.deepcopy(graph),copy.deepcopy(graph),{'graph_capture_failures':0}]}}
    return original,joined,full,{'cases':cases},'binary'

class JoinedModelGateTests(unittest.TestCase):
    def test_complete_retained_metadata(self):
        result=validate_joined_model_gates(*fixtures());self.assertEqual(result['shared_projection_matrices'],336)
    def test_missing_stale_or_mismatched_evidence_blocks_timing(self):
        for change in ('binary','status','missing_state','coverage','hash','inputs','physical','full_binary','token','norm','request','full_repeat','option','native_count','fallback','capture'):
            original,joined,full,golden,binary=fixtures()
            if change=='binary':joined['library_sha256']='stale'
            elif change=='status':joined['status']='starting'
            elif change=='missing_state':joined['cases'].pop()
            elif change=='coverage':joined['cases'][0]['chunks'].pop()
            elif change=='hash':joined['cases'][0]['chunks'][0]['sha256']='4'*64
            elif change=='inputs':joined['inputs']['text']['sha256']='changed'
            elif change=='physical':joined['pool']['compressed_projection_weight_bytes']+=4
            elif change=='full_binary':full['library_sha256']='stale'
            elif change=='token':full['cases'][0]['generated_ids']=[11,99]
            elif change=='norm':full['cases'][0]['normalized_sha256']='different'
            elif change=='request':full['cases'][0]['request_sha256']='different'
            elif change=='full_repeat':full['cases'].pop()
            elif change=='option':full['decoder_final']['packed_profiles']['joined_mlp_projections']=False
            elif change=='native_count':full['decoder_final']['graphs'][0]['native_nvfp4_projection_weights']=192
            elif change=='fallback':full['decoder_final']['graphs'][0]['external_bf16_weight_inputs']=1
            else:full['decoder_final']['graphs'][0]['graph_capture_failures']=1
            with self.subTest(change=change),self.assertRaises(ValueError):validate_joined_model_gates(original,joined,full,golden,binary)
