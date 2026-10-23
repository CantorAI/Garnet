import copy
import hashlib
import json
from pathlib import Path
import unittest
from preflight import audit


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parent/'evidence/base-config.json').read_text(encoding='utf-8-sig'))

    def test_real_checkpoint_and_cache_arithmetic(self):
        r = audit(self.config)
        self.assertEqual(r['cache_estimate_bytes']['attention_kv'], 536870912)
        self.assertEqual(r['cache_estimate_bytes']['recurrent_fp32'], 150994944)
        self.assertFalse(r['garnet_27b_runnable'])
        self.assertEqual(audit(self.config, 4)['cache_estimate_bytes']['attention_kv'], 4*536870912)

    def test_pinned_evidence_integrity(self):
        root = Path(__file__).parent/'evidence'
        manifest = json.loads((root/'checkpoints.json').read_text(encoding='utf-8-sig'))
        for item in manifest['checkpoints']:
            self.assertEqual(len(item['revision']),40)
            self.assertEqual(hashlib.sha256((root/item['file']).read_bytes()).hexdigest(),item['sha256'])

    def test_rejects_wrong_vision_and_small_model(self):
        for section, key, value in [('vision_config','depth',24), ('text_config','hidden_size',2560),
                                    ('text_config','tie_word_embeddings',True)]:
            c = copy.deepcopy(self.config)
            c[section][key] = value
            with self.assertRaises(ValueError): audit(c)

    def test_rejects_wrong_schedule_and_capacity(self):
        c = copy.deepcopy(self.config)
        c['text_config']['layer_types'][0] = 'full_attention'
        with self.assertRaises(ValueError): audit(c)
        for batch, context in [(0,4096),(1,4097),(1,0)]:
            with self.assertRaises(ValueError): audit(self.config,batch,context)

    def test_quantization_is_never_support_claim(self):
        c = copy.deepcopy(self.config)
        c['quantization_config'] = {'quant_method':'modelopt','quant_algo':'MIXED_PRECISION'}
        self.assertFalse(audit(c)['garnet_27b_runnable'])
        c['quantization_config'] = {'quant_method':'unknown'}
        self.assertEqual(audit(c)['format'],'unknown')


if __name__ == '__main__': unittest.main()
