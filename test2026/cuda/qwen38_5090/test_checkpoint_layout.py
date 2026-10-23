"""Contract and quantized-header tests; synthetic fixtures are not GPU evidence."""
import copy, hashlib, json, unittest
from pathlib import Path
from preflight import audit
from inspect_nvfp4 import validate_projection, audit_headers
from xModel.qwen38.vl_27b.model import projection_shapes, validate_config, validate_cuda_profile, validate_image_processor


class LayoutTest(unittest.TestCase):
    def test_pinned_image_processor_and_bounds(self):
        processor=json.loads((Path(__file__).parent/'evidence/nvfp4-preprocessor-config.json').read_text())
        self.assertEqual(validate_image_processor(processor),(65536,1048576))
        changed=copy.deepcopy(processor);changed['size']['shortest_edge']=1024
        with self.assertRaises(ValueError):validate_image_processor(changed)
        changed=copy.deepcopy(processor);changed['image_mean']=[0,0,0]
        with self.assertRaises(ValueError):validate_image_processor(changed)
        for maximum in [65535,1048577]:
            with self.assertRaises(ValueError):validate_image_processor(processor,maximum)

    def setUp(self):
        self.config = json.loads((Path(__file__).parent/'evidence/base-config.json').read_text())

    def test_native_profile_rejects_unquantized_and_unvalidated_capacity(self):
        with self.assertRaises(ValueError):validate_cuda_profile(self.config,4096)
        mixed=json.loads((Path(__file__).parent/'evidence/nvfp4-config.json').read_text())
        self.assertTrue(validate_cuda_profile(mixed,4096))
        for capacity,batch in [(4097,1),(4096,4),(0,1)]:
            with self.assertRaises(ValueError):validate_cuda_profile(mixed,capacity,batch)

    def test_rejects_unsupported_vision_math(self):
        for key,value in [('hidden_act','gelu'),('in_channels',1),('num_position_embeddings',1024)]:
            changed=copy.deepcopy(self.config);changed['vision_config'][key]=value
            with self.assertRaises(ValueError):validate_config(changed)

    def test_architecture_and_distinct_gated_shapes(self):
        shapes = projection_shapes(self.config)
        self.assertEqual(len(shapes),497)
        self.assertEqual(shapes['model.language_model.layers.0.linear_attn.in_proj_qkv'],(10240,5120))
        self.assertEqual(shapes['model.language_model.layers.3.self_attn.q_proj'],(12288,5120))
        self.assertEqual(shapes['lm_head'],(248320,5120))

    def test_pinned_actual_tensor_headers(self):
        root = Path(__file__).parent/'evidence'
        quant = json.loads((root/'nvfp4-config.json').read_text())
        evidence = root/'nvfp4-headers'
        report = json.loads((evidence/'header-audit.json').read_text())
        headers = {}
        for source in report['sources']:
            data = (evidence/source['file']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),source['header_sha256'])
            headers[source['file'].removesuffix('.header.json')] = json.loads(data)
        index = json.loads((evidence/'model.safetensors.index.json').read_text())
        checked = audit_headers(quant,index,headers)
        self.assertEqual(checked['projection_count'],497)
        self.assertEqual(checked['algorithms'],{'NVFP4':193,'FP8':208,'BF16':96})

    def test_rejects_text_only_or_wrong_rope(self):
        for key, value in [('language_model_only',True),('image_token_id',1)]:
            c = copy.deepcopy(self.config); c[key] = value
            with self.assertRaises(ValueError): validate_config(c)
        self.config['text_config']['rope_parameters']['mrope_interleaved'] = False
        with self.assertRaises(ValueError): audit(self.config)

    def test_nvfp4_is_packed_and_double_scaled(self):
        def t(dtype,shape,size): return dict(dtype=dtype,shape=shape,data_offsets=[0,size])
        header = {'p.weight':t('U8',[8,16],128),'p.weight_scale':t('F8_E4M3',[8,2],16),
                  'p.weight_scale_2':t('F32',[],4),'p.input_scale':t('F32',[],4)}
        self.assertEqual(validate_projection(header,'p',(8,32),'NVFP4')['algorithm'],'NVFP4')
        for key in ('p.weight_scale','p.weight_scale_2','p.input_scale'):
            broken = copy.deepcopy(header); del broken[key]
            with self.assertRaises(ValueError): validate_projection(broken,'p',(8,32),'NVFP4')
        broken = copy.deepcopy(header); broken['p.weight']['shape'] = [8,32]
        with self.assertRaises(ValueError): validate_projection(broken,'p',(8,32),'NVFP4')
        broken = copy.deepcopy(header); broken['p.weight_scale']['dtype'] = 'U8'
        with self.assertRaises(ValueError): validate_projection(broken,'p',(8,32),'NVFP4')


if __name__ == '__main__': unittest.main()
