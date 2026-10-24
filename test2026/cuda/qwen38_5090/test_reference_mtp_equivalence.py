import copy, unittest
from reference_mtp_equivalence import compare_pair


class ReferenceMTPGates(unittest.TestCase):
    def test_full_ids_required(self):
        a={'engine':'vllm','key':'text','semantic_request_sha256':'same','prompt_ids':[1,2],'generated_ids':[3,4,5]}
        self.assertTrue(compare_pair(a,copy.deepcopy(a)))
        for field,value in [('prompt_ids',[1,9]),('generated_ids',[3,4,6]),('generated_ids',[])]:
            b=copy.deepcopy(a);b[field]=value;self.assertFalse(compare_pair(a,b))

    def test_engine_and_request_identity_required(self):
        a={'engine':'vllm','key':'text','semantic_request_sha256':'same','prompt_ids':[1],'generated_ids':[2]}
        for field,value in [('engine','sglang'),('key','image'),('semantic_request_sha256','different')]:
            b=copy.deepcopy(a);b[field]=value
            with self.assertRaises(ValueError):compare_pair(a,b)


if __name__=='__main__':unittest.main()
