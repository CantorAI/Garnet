import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from inspect_int4 import audit,validate

class INT4LayoutTests(unittest.TestCase):
    def test_packed_factory_rejects_unsupported_profile_before_gpu_import(self):
        from xModel.qwen38.vl_27b.model import create_cuda_int4_packed_decoder
        config=json.loads((Path(__file__).parent/'evidence/int4-headers/config.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            for chunk in (True,1,32,4.0):
                with self.subTest(chunk=chunk),self.assertRaises(ValueError):create_cuda_int4_packed_decoder(folder,config,4096,chunk)
            with self.assertRaisesRegex(ValueError,'shard index'):create_cuda_int4_packed_decoder(folder,config)
            bad=copy.deepcopy(config);bad['quantization_config']['quant_method']='modelopt'
            with self.assertRaises(ValueError):create_cuda_int4_packed_decoder(folder,bad)
    def test_primary_headers_show_packed_linear_attention(self):
        root=Path(__file__).parent/'evidence/int4-headers';report=json.loads((root/'header-audit.json').read_text());headers={}
        for source in report['sources']:
            raw=(root/(source['file']+'.header.json')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),source['header_sha256']);headers[source['file']]=json.loads(raw)
        records=audit(json.loads((root/'config.json').read_text()),json.loads((root/'model.safetensors.index.json').read_text()),headers)
        self.assertEqual(sum(x['algorithm']=='INT4_W4A16_G128' for x in records),496)
        self.assertEqual(sum(x['algorithm']=='BF16' for x in records),1)
        self.assertTrue(all(x['algorithm']=='INT4_W4A16_G128' for x in records if '.linear_attn.' in x['prefix']))
        shapes=json.loads((root/'model.safetensors.shapes.json').read_text())
        for x in records:
            if x['algorithm']=='INT4_W4A16_G128':self.assertEqual(shapes[x['prefix']+'.weight_shape'],x['logical_shape'])
    def test_rejects_wrong_packed_geometry_or_scale_dtype(self):
        header={'w':{'shape':[16,16],'dtype':'I32','data_offsets':[0,1024]}}
        validate(header,'w',[16,16],{'I32'})
        with self.assertRaises(ValueError):validate(header,'w',[16,8],{'I32'})
        with self.assertRaises(ValueError):validate(header,'w',[16,16],{'BF16'})

if __name__=='__main__':unittest.main()
