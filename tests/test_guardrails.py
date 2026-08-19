import pytest

from dark_factory.guardrails.budget import BudgetExceededError, TokenBudgetTracker
from dark_factory.guardrails.loop_tracker import IterationLoopTracker, LoopLimitExceededError
from dark_factory.llm.types import TokenUsage
from dark_factory.pricing import ModelPrice, PriceBook


def _price_book(input_per_mtok=1.0, output_per_mtok=5.0) -> PriceBook:
    return PriceBook(
        _entries={
            "test:model": ModelPrice(
                input_per_mtok_usd=input_per_mtok, output_per_mtok_usd=output_per_mtok
            )
        }
    )


def test_budget_tracker_accumulates_spend():
    tracker = TokenBudgetTracker(max_budget_usd=1.0, price_book=_price_book())
    tracker.record(
        "agent-a",
        provider="test",
        model="model",
        usage=TokenUsage(input_tokens=1000, output_tokens=1000),
    )
    assert tracker.spent_usd > 0
    assert tracker.remaining_usd == pytest.approx(1.0 - tracker.spent_usd)


def test_budget_tracker_raises_when_ceiling_breached():
    tracker = TokenBudgetTracker(max_budget_usd=0.001, price_book=_price_book())
    with pytest.raises(BudgetExceededError):
        tracker.record(
            "agent-a",
            provider="test",
            model="model",
            usage=TokenUsage(input_tokens=100_000, output_tokens=100_000),
        )
    # Rejected spend must not be charged.
    assert tracker.spent_usd == 0


def test_budget_tracker_respects_per_agent_ceiling():
    tracker = TokenBudgetTracker(
        max_budget_usd=10.0, price_book=_price_book(), max_usd_per_agent={"cheap": 0.001}
    )
    with pytest.raises(BudgetExceededError):
        tracker.record(
            "cheap",
            provider="test",
            model="model",
            usage=TokenUsage(input_tokens=100_000, output_tokens=100_000),
        )


def test_budget_tracker_mock_provider_is_free():
    tracker = TokenBudgetTracker(max_budget_usd=0.0, price_book=PriceBook.builtin())
    cost = tracker.record(
        "agent-a", provider="mock", model="mock-default", usage=TokenUsage(input_tokens=10**9)
    )
    assert cost == 0.0


def test_loop_tracker_allows_up_to_max_iterations():
    tracker = IterationLoopTracker(loop_name="build-test-fix", max_iterations=3)
    assert tracker.step() == 1
    assert tracker.step() == 2
    assert tracker.step() == 3
    assert tracker.is_exhausted


def test_loop_tracker_raises_past_max_iterations():
    tracker = IterationLoopTracker(loop_name="build-test-fix", max_iterations=2)
    tracker.step()
    tracker.step()
    with pytest.raises(LoopLimitExceededError):
        tracker.step()


def test_loop_tracker_reset():
    tracker = IterationLoopTracker(loop_name="build-test-fix", max_iterations=1)
    tracker.step()
    assert tracker.is_exhausted
    tracker.reset()
    assert not tracker.is_exhausted
    assert tracker.step() == 1
