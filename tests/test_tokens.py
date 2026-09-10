"""Token counting, truncation, and budget packing."""

from doc_agent.tokens import Budget, count_tokens, pack_to_budget, truncate_to_tokens


def test_count_tokens_is_not_character_count():
    """The whole reason for tiktoken: tokens and characters diverge."""
    dense = "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis"
    sparse = "a a a a a a a a a a a a a a a a a a a a a a a a a"

    assert len(dense) < len(sparse) * 3
    assert count_tokens(dense) < count_tokens(sparse)


def test_truncate_respects_token_limit():
    text = "word " * 500

    assert count_tokens(truncate_to_tokens(text, 50)) == 50


def test_truncate_leaves_short_text_alone():
    assert truncate_to_tokens("short", 100) == "short"


def test_budget_subtracts_reservations():
    budget = Budget(context_window=8000, reserved_for_answer=1000, reserved_for_prompt=500)

    assert budget.available_for_context == 6500


def test_budget_never_goes_negative():
    budget = Budget(context_window=100, reserved_for_answer=200, reserved_for_prompt=200)

    assert budget.available_for_context == 0


def test_packing_keeps_the_highest_ranked_items_that_fit():
    items = ["alpha " * 30, "beta " * 30, "gamma " * 30]
    budget = count_tokens(items[0]) + count_tokens(items[1])

    result = pack_to_budget(items, budget)

    assert result.included == items[:2]
    assert result.dropped == 1
    assert result.used_tokens <= budget


def test_packing_skips_an_oversized_item_and_keeps_going():
    """A later small item should still fit after a big one is skipped."""
    items = ["huge " * 200, "tiny"]

    result = pack_to_budget(items, budget=50)

    assert result.included == ["tiny"]
    assert result.dropped == 1


def test_packing_reports_totals():
    result = pack_to_budget(["a", "b", "c"], budget=1000)

    assert result.total == 3
    assert result.dropped == 0
