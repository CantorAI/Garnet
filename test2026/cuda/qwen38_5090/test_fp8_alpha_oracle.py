"""FP8 diagnostic arithmetic must round combined alpha in F32 before BF16 output."""
import unittest
import numpy as np
import prepare_real_layer_probe as arithmetic

class AlphaOracleTest(unittest.TestCase):
    def test_combined_alpha_tie_rounds_to_even_bf16(self):
        class Checkpoint:
            def tensor(self,name):
                raw=np.zeros(16,dtype=np.uint8);raw[:2]=[0xd7,0x58] # -15,+16
                return raw,{'dtype':'F8_E4M3','shape':[1,16]}
            def scalar(self,name):
                return float(np.float32(10.0390625 if name.endswith('input_scale') else .1))
        previous=arithmetic.FP8_ALPHA_AFTER_DOT
        try:
            arithmetic.FP8_ALPHA_AFTER_DOT=True
            x=np.zeros(16,dtype=np.float32);x[:2]=161 # BF16 inputs quantize to16,16
            # Raw dot is16; combined F32 alpha is1.00390625. The16.0625
            # midpoint rounds to16.0 (even), not16.125, at the BF16 boundary.
            result=arithmetic.project(Checkpoint(),'p',x)
            self.assertEqual(float(result[0]),16.)
        finally:arithmetic.FP8_ALPHA_AFTER_DOT=previous

if __name__=='__main__':unittest.main()
