import json,unittest
from native_worker_io import requests,emit,MAX_FRAME_BYTES

class WorkerIOTests(unittest.TestCase):
    def reader(self,blocks):
        it=iter(blocks)
        return lambda:next(it,b'')
    def test_fragmented_multiple_frames_and_eof(self):
        self.assertEqual(list(requests(self.reader([b'{"id":',b'1}\n{"id":2}\r\n']))),[{'id':1},{'id':2}])
        with self.assertRaises(ValueError):list(requests(self.reader([b'{"id":1}'])))
        with self.assertRaises(ValueError):list(requests(self.reader([b'[]\n'])))
    def test_oversized_frames_rejected_before_dispatch(self):
        with self.assertRaises(ValueError):list(requests(self.reader([b' '* (MAX_FRAME_BYTES+1)])))
        with self.assertRaises(ValueError):list(requests(self.reader([b' '* (MAX_FRAME_BYTES+1)+b'\n'])))
    def test_partial_writes_preserve_actual_event_and_fail_on_closed_pipe(self):
        parts=[]
        def write(raw):parts.append(raw[:3]);return len(raw[:3])
        emit({'event':'token','token_id':248046},write)
        self.assertEqual(json.loads(b''.join(parts)),{'event':'token','token_id':248046})
        self.assertTrue(b''.join(parts).endswith(b'\n'))
        with self.assertRaises(OSError):emit({},lambda raw:0)
if __name__=='__main__':unittest.main()
