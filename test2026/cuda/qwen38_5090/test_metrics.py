"""Synthetic arithmetic fixtures only, not hardware benchmark results."""
import unittest
from metrics import summarize, meets_two_x


def row(s=0, first=1, last=3, done=4, n=5):
    return dict(status='completed',submitted=s,first_token=first,last_token=last,completed=done,generated_tokens=n)


class MetricsTest(unittest.TestCase):
    def test_excludes_first_token_from_decode_and_counts_cohort_wall(self):
        r = summarize([row(),row(s=0.5,first=1.5,last=3.5,done=4.5)])
        self.assertEqual(r['decode_median_tokens_per_second'],2)
        self.assertAlmostEqual(r['output_tokens_per_second'],10/4.5)
        self.assertEqual(r['ttft_median_ms'],1000)

    def test_single_token_decode_undefined(self):
        r = summarize([row(first=1,last=1,n=1)])
        self.assertIsNone(r['decode_median_tokens_per_second'])

    def test_speculative_first_arrival_batch_is_excluded_in_full(self):
        result=summarize([dict(row(n=5),first_arrival_batch_tokens=3)])
        self.assertEqual(result['decode_median_tokens_per_second'],1)
        self.assertEqual(result['generated_tokens'],5)
        self.assertEqual(result['ttft_median_ms'],1000)
        self.assertEqual(result['output_tokens_per_second'],1.25)
        for count in (0,6,True):
            with self.assertRaises(ValueError):summarize([dict(row(n=5),first_arrival_batch_tokens=count)])

    def test_failures_and_bad_events_cannot_claim_speed(self):
        for r in [dict(row(),status='error'),row(first=3,last=2),row(first=1,last=1),row(n=0),row(first=float('nan'))]:
            with self.assertRaises(ValueError): summarize([r])

    def test_two_x_requires_both_metrics_and_faster_engine(self):
        def metrics(v): return dict(decode_median_tokens_per_second=v,output_tokens_per_second=v)
        self.assertTrue(meets_two_x(metrics(4),metrics(1),metrics(2)))
        self.assertFalse(meets_two_x(metrics(3.9),metrics(1),metrics(2)))
        self.assertFalse(meets_two_x(dict(metrics(4),output_tokens_per_second=3),metrics(1),metrics(2)))
        self.assertFalse(meets_two_x(metrics(4),metrics(1),metrics(None)))


if __name__ == '__main__': unittest.main()
