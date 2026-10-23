"""Verify paper logs preserve both successful and failed commands across runs."""
import json, sys, tempfile, unittest
from pathlib import Path
from run_logged import run


class LedgerTest(unittest.TestCase):
    def test_append_preserves_failure_exit_and_raw_output(self):
        parent = Path(__file__).resolve().parents[3]/'out/qwen38-ledger-tests'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            self.assertEqual(run([sys.executable,'-c',"print('CONTROL_FIXTURE')"],root,'correctness','control'),0)
            self.assertEqual(run([sys.executable,'-c',"print('FAILED_FIXTURE'); raise SystemExit(7)"],root,'correctness','failed'),7)
            records = [json.loads(line) for line in (root/'commands.jsonl').read_text().splitlines()]
            finished = [r for r in records if r['event']=='finished']
            self.assertEqual([r['exit_code'] for r in finished],[0,7])
            raw = (root/'commands.log').read_bytes()
            for item, marker in zip(finished,(b'CONTROL_FIXTURE',b'FAILED_FIXTURE')):
                self.assertIn(marker,raw[item['raw_start_offset']:item['raw_end_offset']])
            self.assertEqual(len({r['id'] for r in finished}),2)

    def test_cannot_relabel_process_duration_as_benchmark(self):
        with self.assertRaises(ValueError): run([],Path('.'),'benchmark','invalid')


if __name__ == '__main__': unittest.main()
