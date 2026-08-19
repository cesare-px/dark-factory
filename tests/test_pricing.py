import pytest

from dark_factory.llm.types import TokenUsage
from dark_factory.pricing import ModelNotPricedError, ModelPrice, PriceBook, normalize_model_id


def test_normalize_strips_date_suffix():
    assert normalize_model_id("claude-sonnet-5-20260215") == "claude-sonnet-5"
    assert normalize_model_id("claude-sonnet-5") == "claude-sonnet-5"


def test_normalize_strips_bedrock_region_and_version():
    assert normalize_model_id("us.anthropic.claude-3-sonnet-v1:0") == "anthropic.claude-3-sonnet"


def test_price_for_exact_match():
    book = PriceBook.builtin()
    price = book.price_for("anthropic", "claude-sonnet-5")
    assert price.input_per_mtok_usd == 2.00
    assert price.output_per_mtok_usd == 10.00


def test_price_for_normalized_match():
    book = PriceBook.builtin()
    price = book.price_for("anthropic", "claude-sonnet-5-20260301")
    assert price.input_per_mtok_usd == 2.00


def test_price_for_glob_pattern():
    book = PriceBook(
        _patterns={"custom:ft-*": ModelPrice(input_per_mtok_usd=1.0, output_per_mtok_usd=2.0)}
    )
    price = book.price_for("custom", "ft-my-finetune-v3")
    assert price.input_per_mtok_usd == 1.0


def test_free_by_default_providers_need_no_pricing():
    book = PriceBook.builtin()
    for provider in ("mock", "openai_compatible", "ollama"):
        price = book.price_for(provider, "whatever-model")
        assert price.input_per_mtok_usd == 0.0
        assert price.output_per_mtok_usd == 0.0


def test_unknown_paid_model_raises_by_default():
    book = PriceBook.builtin()
    with pytest.raises(ModelNotPricedError):
        book.price_for("openai", "gpt-does-not-exist")


def test_unknown_model_policy_warn_fallback():
    book = PriceBook.builtin(on_unknown_model="warn_fallback")
    price = book.price_for("openai", "gpt-does-not-exist")
    assert price.input_per_mtok_usd > 0  # conservative synthetic price, not $0


def test_unknown_model_policy_zero():
    book = PriceBook.builtin(on_unknown_model="zero")
    price = book.price_for("openai", "gpt-does-not-exist")
    assert price.input_per_mtok_usd == 0.0


def test_with_overrides_takes_precedence_over_builtin():
    book = PriceBook.builtin().with_overrides(
        {"anthropic:claude-sonnet-5": {"input_per_mtok_usd": 999.0, "output_per_mtok_usd": 999.0}}
    )
    price = book.price_for("anthropic", "claude-sonnet-5")
    assert price.input_per_mtok_usd == 999.0
    assert price.source == "config"


def test_cost_usd_math():
    price = ModelPrice(input_per_mtok_usd=2.0, output_per_mtok_usd=10.0)
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    assert price.cost_usd(usage) == pytest.approx(2.0 + 5.0)


def test_cost_usd_uses_default_cache_multipliers_when_unset():
    price = ModelPrice(input_per_mtok_usd=10.0, output_per_mtok_usd=50.0)
    usage = TokenUsage(cached_input_tokens=1_000_000, cache_write_tokens=1_000_000)
    # cached defaults to 0.1x input, write defaults to 1.25x input
    assert price.cost_usd(usage) == pytest.approx(1.0 + 12.5)
