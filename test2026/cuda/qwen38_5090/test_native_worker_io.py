import json,unittest
from native_worker_io import requests,emit,MAX_FRAME_BYTES,transport_text

class WorkerIOTests(unittest.TestCase):
    def test_native_byte_prefix_normalization_matches_utf8_codec(self):
        class NativeText:
            def __init__(self,raw):self.raw=raw
            def encode(self,encoding):return self.raw
        vectors=[b'\xf0\x9f\x94\xb5',b'\xe4\xb8\xad',b'\xc3\xa9',b'\xf0\x9fA',
            b'\xed\xa0\x80',b'\xe0\x80\x80',b'\xf4\x90\x80\x80',b'\xff\x80',b'\xe2\x82\xac']
        for raw in vectors:
            for count in range(len(raw)+1):
                prefix=b'prefix '+raw[:count]
                self.assertEqual(transport_text(NativeText(prefix)),prefix.decode('utf8','replace'))
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
    def test_unicode_text_partial_writes_preserve_ids_and_caller(self):
        value={'event':'token','text':'Here\u2019s \u4e2d\u6587 \U0001f535\n"quoted"','token_id':91786,'sampled_tokens':2}
        before=dict(value);parts=[]
        def write(raw):parts.append(raw[:3]);return len(raw[:3])
        emit(value,write)
        self.assertEqual(json.loads(b''.join(parts)),before)
        self.assertEqual(value,before)
        self.assertEqual(b''.join(parts).count(b'\n'),1)
if __name__=='__main__':unittest.main()
