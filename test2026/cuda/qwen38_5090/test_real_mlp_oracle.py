import json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from prepare_real_mlp_probe import bf16,fp8_table,nearest,Checkpoint,project

class RealMLPOracleTests(unittest.TestCase):
    def test_ties_even_and_saturation(self):
        table=fp8_table()
        self.assertEqual(table[126],448.)
        self.assertEqual(nearest(np.array([1.0625,1.1875,1000.]),table).tolist(),[56,58,126])
        self.assertEqual(bf16([1.00390625,1.01171875]).tolist(),[1.,1.015625])
    def test_independently_verified_native_identity_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            subprocess.run([sys.executable,str(Path(__file__).with_name('make_mlp_fixture.py')),tmp],check=True)
            with (root/'model.safetensors').open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in header}}))
            checkpoint=Checkpoint(root);prefix='model.language_model.layers.0.mlp.'
            x=np.full(256,1.5,dtype=np.float32)
            gate=project(checkpoint,prefix+'gate_proj',x);up=project(checkpoint,prefix+'up_proj',x)
            self.assertTrue(np.array_equal(gate,x))
            result=project(checkpoint,prefix+'down_proj',(gate/(1+np.exp(-gate)))*up)
            self.assertEqual(result.tolist(),[1.9296875]*256)

if __name__=='__main__':unittest.main()
