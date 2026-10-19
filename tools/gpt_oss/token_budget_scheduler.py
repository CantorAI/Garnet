"""Pure planner for a bounded vLLM-inspired token-budget scheduling step.

This module does not execute Garnet kernels. It models one scheduler decision so
that its token budget, FIFO order, request ownership, and peer-row bounds can be
checked on CPU before any dynamic-batch serving integration is attempted.
"""
from dataclasses import dataclass


DEFAULT_TOKEN_BUDGET = 8192
DEFAULT_MAX_SEQUENCES = 224
DEFAULT_PREFILL_CHUNK = 36
DEFAULT_PREFILL_COHORT = 112
DEFAULT_MAX_PEER_ROWS = 4096


@dataclass(frozen=True)
class QueuedPrefill:
    request_id: str
    remaining_tokens: int
    input_offset: int = 0


@dataclass(frozen=True)
class PrefillSlice:
    request_id: str
    input_offset: int
    real_tokens: int
    padded_tokens: int


@dataclass(frozen=True)
class TokenBudgetPlan:
    priority_request_ids: tuple[str, ...]
    prefill_cohorts: tuple[tuple[PrefillSlice, ...], ...]
    token_budget: int
    scheduled_tokens: int
    live_sequences_after_admission: int
    peer_rows_per_prefill_cohort: tuple[int, ...]


def plan_token_budget_iteration(
    active_priority_request_ids,
    queued_prefills,
    *,
    token_budget=DEFAULT_TOKEN_BUDGET,
    max_sequences=DEFAULT_MAX_SEQUENCES,
    prefill_chunk=DEFAULT_PREFILL_CHUNK,
    prefill_cohort=DEFAULT_PREFILL_COHORT,
    max_peer_rows=DEFAULT_MAX_PEER_ROWS,
    max_new_prefills=None,
):
    """Plan decodes first, then FIFO prefill slices within both hard budgets.

    Each new request receives at most one prefill slice in this scheduler
    iteration. Two disjoint 112x36 cohorts can therefore use 8,064 of an
    8,192-token budget, while every individual peer-group dispatch expands to
    at most 4,032 rows. ``max_new_prefills`` can stage a smaller FIFO admission
    wave to overlap request lifetimes across iterations. The caller owns KV
    allocation and state transitions.
    """
    if min(token_budget, max_sequences, prefill_chunk, prefill_cohort,
           max_peer_rows) <= 0:
        raise ValueError("scheduler limits must be positive")
    if prefill_chunk > max_peer_rows:
        raise ValueError("prefill chunk exceeds peer-row capacity")
    if prefill_cohort * prefill_chunk > max_peer_rows:
        raise ValueError("prefill cohort exceeds peer-row capacity")
    if max_new_prefills is not None and max_new_prefills <= 0:
        raise ValueError("new-prefill admission limit must be positive")

    active = tuple(active_priority_request_ids)
    queued = tuple(queued_prefills)
    ids = [request_id for request_id in active]
    if any(not isinstance(request_id, str) or not request_id for request_id in ids):
        raise ValueError("active request IDs must be non-empty strings")
    if len(ids) > max_sequences:
        raise ValueError("active requests exceed the live-sequence limit")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate active request ID")

    queued_ids = []
    for item in queued:
        if not isinstance(item, QueuedPrefill):
            raise ValueError("queued entries must be QueuedPrefill records")
        if not isinstance(item.request_id, str) or not item.request_id:
            raise ValueError("queued request IDs must be non-empty strings")
        if item.remaining_tokens <= 0 or item.input_offset < 0:
            raise ValueError("queued prompt offsets and remaining tokens are invalid")
        queued_ids.append(item.request_id)
    if len(queued_ids) != len(set(queued_ids)) or set(ids).intersection(queued_ids):
        raise ValueError("request may appear only once across active and queued state")

    decode_count = min(len(active), token_budget, max_peer_rows)
    decode_ids = active[:decode_count]
    used = decode_count
    # Existing active sequences retain their slots even when a constrained
    # budget prevents scheduling every decode in this step.
    slots = max_sequences - len(active)
    if decode_count < len(active):
        slots = 0

    selected = []
    prefill_limit = min(slots, len(queued),
                        max_new_prefills if max_new_prefills is not None else len(queued))
    for item in queued:
        if len(selected) >= prefill_limit:
            break
        real_tokens = min(item.remaining_tokens, prefill_chunk)
        if used + real_tokens > token_budget:
            break
        selected.append(PrefillSlice(item.request_id, item.input_offset,
                                     real_tokens, prefill_chunk))
        used += real_tokens

    cohorts = tuple(tuple(selected[i:i + prefill_cohort])
                    for i in range(0, len(selected), prefill_cohort))
    row_counts = tuple(len(cohort) * prefill_chunk for cohort in cohorts)
    if any(rows > max_peer_rows for rows in row_counts):
        raise AssertionError("planner emitted a prefill dispatch above the peer-row cap")
    live = len(active) + len(selected)
    if live > max_sequences:
        raise AssertionError("planner admitted too many live sequences")
    if used > token_budget:
        raise AssertionError("planner exceeded its aggregate token budget")

    return TokenBudgetPlan(tuple(decode_ids), cohorts, token_budget, used,
                           live, row_counts)
