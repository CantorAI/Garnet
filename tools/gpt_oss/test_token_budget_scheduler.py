import unittest

from token_budget_scheduler import (
    QueuedPrefill,
    plan_token_budget_iteration,
)


class TokenBudgetSchedulerTests(unittest.TestCase):
    def test_two_disjoint_cohorts_approach_vllm_token_budget_safely(self):
        queued = [QueuedPrefill(f"q{i:03d}", 2005) for i in range(224)]
        plan = plan_token_budget_iteration((), queued)

        self.assertEqual(plan.priority_request_ids, ())
        self.assertEqual([len(group) for group in plan.prefill_cohorts], [112, 112])
        self.assertEqual(plan.peer_rows_per_prefill_cohort, (4032, 4032))
        self.assertEqual(plan.scheduled_tokens, 8064)
        self.assertLessEqual(plan.scheduled_tokens, plan.token_budget)
        self.assertEqual(plan.live_sequences_after_admission, 224)
        self.assertEqual(
            [item.request_id for group in plan.prefill_cohorts for item in group],
            [f"q{i:03d}" for i in range(224)],
        )

    def test_priority_work_is_ordered_before_remaining_prefill_capacity(self):
        active = [f"a{i:03d}" for i in range(112)]
        queued = [QueuedPrefill(f"q{i:03d}", 2005) for i in range(112)]
        plan = plan_token_budget_iteration(active, queued)

        self.assertEqual(plan.priority_request_ids, tuple(active))
        self.assertEqual([len(group) for group in plan.prefill_cohorts], [112])
        self.assertEqual(plan.scheduled_tokens, 112 + 112 * 36)
        self.assertEqual(plan.live_sequences_after_admission, 224)

    def test_fifo_and_partial_prefill_offsets_are_preserved(self):
        queued = [QueuedPrefill("first", 7, 36), QueuedPrefill("second", 80, 0)]
        plan = plan_token_budget_iteration((), queued, max_sequences=2)

        self.assertEqual(len(plan.prefill_cohorts), 1)
        self.assertEqual(
            [(x.request_id, x.input_offset, x.real_tokens, x.padded_tokens)
             for x in plan.prefill_cohorts[0]],
            [("first", 36, 7, 36), ("second", 0, 36, 36)],
        )
        self.assertEqual(plan.scheduled_tokens, 43)
        self.assertEqual(plan.peer_rows_per_prefill_cohort, (72,))

    def test_existing_live_sequences_keep_slots_if_priority_budget_is_short(self):
        active = ["a", "b", "c"]
        queued = [QueuedPrefill("q", 12)]
        plan = plan_token_budget_iteration(active, queued, token_budget=2,
                                           max_sequences=4)

        self.assertEqual(plan.priority_request_ids, ("a", "b"))
        self.assertEqual(plan.prefill_cohorts, ())
        self.assertEqual(plan.live_sequences_after_admission, 3)
        self.assertEqual(plan.scheduled_tokens, 2)

    def test_rejects_duplicate_ids_invalid_offsets_and_unsafe_cohorts(self):
        with self.assertRaises(ValueError):
            plan_token_budget_iteration(["x"], [QueuedPrefill("x", 1)])
        with self.assertRaises(ValueError):
            plan_token_budget_iteration([], [QueuedPrefill("x", 0)])
        with self.assertRaises(ValueError):
            plan_token_budget_iteration([], [QueuedPrefill("x", 1, -1)])
        with self.assertRaises(ValueError):
            plan_token_budget_iteration([], [], prefill_cohort=114)
        with self.assertRaises(ValueError):
            plan_token_budget_iteration(["a", "b"], [], max_sequences=1)


if __name__ == "__main__":
    unittest.main()
