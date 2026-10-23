"""Synthetic transport fixtures verify artifact integrity and disk budget gates."""
import hashlib, io, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import download_pinned as downloader


class Response(io.BytesIO):
    status = 200
    headers = {}


class DownloadTest(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[3]/'out/qwen38-download-tests'
        parent.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, expected, body=b'fixture', size=7,model=downloader.MODEL,revision=None):
        manifest = self.root/'transport.json'
        manifest.write_text(json.dumps({'model':model,'revision':revision or downloader.PINS.get(model,('0'*40,0))[0],
            'files':[{'name':'tiny.bin','bytes':size,'sha256':expected,'url':'https://example.invalid/public?expires=1'}]}))
        output = self.root/'model'
        with patch.object(sys,'argv',['download_pinned',str(manifest),str(output)]), \
             patch.object(downloader,'urlopen',side_effect=lambda *a,**k:Response(body)), \
             patch.object(downloader.time,'sleep'):
            downloader.main()
        return output

    def test_exact_bytes_and_provenance_without_signed_url(self):
        output = self.invoke(hashlib.sha256(b'fixture').hexdigest())
        self.assertEqual((output/'tiny.bin').read_bytes(),b'fixture')
        provenance = (output/'download-provenance.json').read_text()
        self.assertNotIn('expires=',provenance)
        self.assertFalse((self.root/'transport.json').exists())

    def test_small_comparison_model_keeps_its_own_provenance(self):
        output=self.invoke(hashlib.sha256(b'fixture').hexdigest(),model='Qwen/Qwen3.5-4B')
        provenance=json.loads((output/'download-provenance.json').read_text())
        self.assertEqual(provenance['model'],'Qwen/Qwen3.5-4B')
        self.assertEqual(provenance['revision'],downloader.PINS['Qwen/Qwen3.5-4B'][0])

    def test_unpinned_revision_or_repository_is_rejected(self):
        with self.assertRaises(ValueError):self.invoke('0'*64,revision='0'*40)
        with self.assertRaises(ValueError):self.invoke('0'*64,model='unverified/model')

    def test_corrupt_bytes_never_publish_final_artifact(self):
        with self.assertRaises(RuntimeError): self.invoke(hashlib.sha256(b'different').hexdigest())
        self.assertFalse((self.root/'model/tiny.bin').exists())
        self.assertFalse((self.root/'model/download-provenance.json').exists())

    def test_oversized_manifest_rejected_before_network(self):
        with self.assertRaises(ValueError): self.invoke('0'*64,size=25*1024**3)


if __name__ == '__main__': unittest.main()
