"""Synthetic parser and independent byte-oracle tests; no GPU qualification."""
import json,struct,tempfile,unittest
from pathlib import Path
from host_embedding_fixture import NAME,prepare

class FixtureTests(unittest.TestCase):
    def checkpoint(self,base,**overrides):
        folder=base/'checkpoint';folder.mkdir()
        # Includes signed zero, smallest BF16 subnormal, negative normal and max finite.
        words=[0x0000,0x8000,0x0001,0x3f80,0xbf80,0x7f7f,0x0080,0x4000]
        entry={'dtype':'BF16','shape':[4,2],'data_offsets':[0,16]};entry.update(overrides)
        h=json.dumps({NAME:entry}).encode();(folder/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+struct.pack('<8H',*words))
        return folder,words
    def test_known_bytes_boundaries_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);checkpoint,words=self.checkpoint(base);out=base/'fixture';d=prepare(checkpoint,out)
            self.assertEqual(d['ids'][:4],[0,3,1,0]);self.assertFalse(d['gpu_qualified'])
            for token,paths in d['rows'].items():
                expected=struct.pack('<2I',*(x*65536 for x in words[int(token)*2:int(token)*2+2]))
                self.assertEqual((out/paths['f32']).read_bytes(),expected)
            with self.assertRaises(FileExistsError):prepare(checkpoint,out)
    def test_invalid_geometry_rejected_before_output(self):
        for overrides in [{'dtype':'F16'},{'shape':[4,3]},{'data_offsets':[0,999]},{'data_offsets':[-1,15]}]:
            with tempfile.TemporaryDirectory() as temp:
                base=Path(temp);checkpoint,_=self.checkpoint(base,**overrides)
                with self.assertRaises(ValueError):prepare(checkpoint,base/'fixture')
                self.assertFalse((base/'fixture').exists())
    def test_index_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);checkpoint,_=self.checkpoint(base)
            (base/'outside.safetensors').write_bytes((checkpoint/'model.safetensors').read_bytes())
            (checkpoint/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{NAME:'../outside.safetensors'}}))
            with self.assertRaisesRegex(ValueError,'escapes'):prepare(checkpoint,base/'fixture')

if __name__=='__main__':unittest.main()
