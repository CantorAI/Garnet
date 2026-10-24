import json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from xModel.qwen35.vl_0_8b.joined_mlp_profile import validate_joined_mlp_profile
from xModel.qwen38.vl_27b.model import create_cuda_shared_compressed_decoder

class JoinedMLPProfileTests(unittest.TestCase):
    def test_admission_fails_before_gpu_for_unqualified_profiles(self):
        for size in (4,128):validate_joined_mlp_profile(True,1,4096,size,True)
        for enabled,batch,capacity,size,external,embedded,joined_inputs in [
            (1,1,4096,4,True,False,False),(True,True,4096,4,True,False,False),
            (True,2,4096,4,True,False,False),(True,1,8192,4,True,False,False),
            (True,1,4096.0,4,True,False,False),(True,1,4096,16,True,False,False),
            (True,1,4096,64,True,False,False),(True,1,4096,4,False,False,False),
            (True,1,4096,4,True,True,False),(True,1,4096,4,True,False,True)]:
            with self.subTest(profile=(enabled,batch,capacity,size,external,embedded,joined_inputs)),self.assertRaises(ValueError):
                validate_joined_mlp_profile(enabled,batch,capacity,size,external,embedded,joined_inputs)
        # The new opt-in must not narrow the pre-existing disabled path.
        validate_joined_mlp_profile(False,8,8192,16,False)

    def test_factory_forwards_optin_and_rejects_wrong_native_counts(self):
        config=json.loads((Path(__file__).parent/'evidence/nvfp4-config.json').read_text());instances=[];corrupt={}
        class FakeDecoder:
            def __init__(self,*args,**kwargs):
                self.kwargs=kwargs;self.closed=False;instances.append(self)
                nv=128 if kwargs['joined_mlp_projections'] else 192;total=nv+208
                pool={'compressed_projection_pool_entries':total,'compressed_nvfp4_entries':nv,'compressed_fp8_entries':208,'compressed_projection_weight_bytes':1024,'compressed_projection_scale_bytes':128}
                stats={'native_nvfp4_projection_weights':nv,'native_fp8_projection_weights':208,'external_bf16_weight_inputs':0,'external_compressed_weight_inputs':total,'external_compressed_weight_bytes':1152};stats.update(corrupt)
                self.io=types.SimpleNamespace(enable_nvfp4_head=lambda enabled:None,memory_json=lambda:json.dumps(pool))
                self.programs={size:types.SimpleNamespace(graph=types.SimpleNamespace(stats_json=lambda:json.dumps(stats))) for size in (4,128)}
            def close(self):self.closed=True
        module=types.SimpleNamespace(CUDAPackedDecoder=FakeDecoder)
        with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}):
            original=create_cuda_shared_compressed_decoder('unused',config,4096)
            candidate=create_cuda_shared_compressed_decoder('unused',config,4096,joined_mlp_projections=True)
            self.assertIs(original.kwargs['joined_mlp_projections'],False);self.assertIs(candidate.kwargs['joined_mlp_projections'],True)
            for field,value in [('native_nvfp4_projection_weights',192),('native_fp8_projection_weights',207),('external_compressed_weight_inputs',400),('external_bf16_weight_inputs',1),('external_compressed_weight_bytes',999)]:
                corrupt.clear();corrupt[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):create_cuda_shared_compressed_decoder('unused',config,4096,joined_mlp_projections=True)
                self.assertTrue(instances[-1].closed)
            with self.assertRaises(ValueError):create_cuda_shared_compressed_decoder('unused',config,4096,prefill_chunk=64,joined_mlp_projections=True)
