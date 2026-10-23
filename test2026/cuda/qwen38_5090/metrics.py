"""Measurement arithmetic over real token event records; no engine invocation."""
import math
import statistics


def summarize(rows):
    """Each row contains monotonic seconds and actual generated token count.

    Warmups must be removed by the caller. last_token means last generated token,
    while completed includes transport and final output processing. Chunk counts
    must never be supplied as generated_tokens.
    """
    if not rows:
        raise ValueError('No measured requests')
    ttft, decode = [], []
    tokens = 0
    start, end = math.inf, -math.inf
    for r in rows:
        if r.get('status') != 'completed':
            raise ValueError('Failed/incomplete requests invalidate comparison; preserve raw errors')
        times = [r[k] for k in ('submitted','first_token','last_token','completed')]
        if not all(isinstance(t,(int,float)) and math.isfinite(t) for t in times):
            raise ValueError('Finite monotonic timestamps required')
        s, first, last, done = times
        n = r['generated_tokens']
        if isinstance(n,bool) or not isinstance(n,int) or n < 1 or not s <= first <= last <= done:
            raise ValueError('Invalid token count or event ordering')
        if n > 1 and last <= first:
            raise ValueError('Multiple tokens require a positive decode interval')
        ttft.append((first-s)*1000)
        if n > 1: decode.append((n-1)/(last-first))
        tokens += n
        start, end = min(start,s), max(end,done)
    if end <= start:
        raise ValueError('Positive cohort wall duration required')
    return {'requests':len(rows), 'generated_tokens':tokens,
            'ttft_median_ms':statistics.median(ttft),
            'decode_median_tokens_per_second':statistics.median(decode) if decode else None,
            'output_tokens_per_second':tokens/(end-start),
            'requests_per_second':len(rows)/(end-start)}


def meets_two_x(garnet, vllm, sglang):
    """Caller must establish hardware/workload/quality equivalence first."""
    for key in ('decode_median_tokens_per_second','output_tokens_per_second'):
        vals = [r.get(key) for r in (garnet,vllm,sglang)]
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v > 0 for v in vals):
            return False
        if vals[0] < 2*max(vals[1:]): return False
    return True
