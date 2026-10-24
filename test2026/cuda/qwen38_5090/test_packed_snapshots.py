import array,math,tempfile,unittest
from pathlib import Path
from compare_packed_snapshots import compare_files
class SnapshotTests(unittest.TestCase):
 def test_small_segment_error_is_not_hidden_by_large_other_state(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=Path(tmp)/'a';b=Path(tmp)/'b'
   a.write_bytes(array.array('f',[.02]+[100.]*1000).tobytes());b.write_bytes(array.array('f',[.01]+[100.]*1000).tobytes())
   self.assertTrue(compare_files(a,b)['passed']);self.assertFalse(compare_files(a,b,0,1)['passed'])
   self.assertTrue(compare_files(a,b,1,1000)['passed'])
   with self.assertRaises(ValueError):compare_files(a,b,1000,2)
 def test_nonfinite_and_truncated_snapshot_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=Path(tmp)/'a';b=Path(tmp)/'b';a.write_bytes(array.array('f',[float('nan')]).tobytes());b.write_bytes(array.array('f',[1.]).tobytes())
   with self.assertRaises(ValueError):compare_files(a,b)
   a.write_bytes(b'123')
   with self.assertRaises(ValueError):compare_files(a,b)
if __name__=='__main__':unittest.main()
