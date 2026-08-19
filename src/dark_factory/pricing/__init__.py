"""Per-model USD pricing, decoupled from any provider SDK or config format.

Prices are shipped, diffable data (prices.yml), not Python constants, and are
layerable with user overrides from FactoryConfig. Lookup never guesses: an
unpriced model on a paid provider is a hard error by default, because
defaulting to $0 would silently disable the budget ceiling.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import yaml

from dark_factory.llm.types import TokenUsage

UnknownModelPolicy = Literal["error", "warn_fallback", "zero"]

# Self-hosted / offline providers have no per-token API cost.
FREE_BY_DEFAULT_PROVIDERS = frozenset({"mock", "echo", "openai_compatible", "ollama"})

_DATE_SUFFIX = re.compile(r"-\d{8}$")
_BEDROCK_REGION_PREFIX = re.compile(r"^(us|eu|apac)\.")
_BEDROCK_VERSION_SUFFIX = re.compile(r"-v\d+:\d+$")


class ModelNotPricedError(RuntimeError):
    """A paid provider's model has no known price and no override was given."""

    def __init__(self, provider: str, model: str) -> None:
        """Build the error with a copy-pasteable config snippet to fix it."""
        self.provider = provider
        self.model = model
        key = f"{provider}:{model}"
        super().__init__(
            f"no price for '{key}'.\n"
            f"  Add to .dark-factory.yml:\n"
            f'      pricing:\n        "{key}":\n'
            f"          input_per_mtok_usd: 0.00\n"
            f"          output_per_mtok_usd: 0.00\n"
            f"  Or set: pricing.on_unknown_model: warn_fallback"
        )


def normalize_model_id(model: str) -> str:
    """Strip volatile suffixes so a dated snapshot matches its price entry."""
    normalized = _DATE_SUFFIX.sub("", model)
    normalized = _BEDROCK_REGION_PREFIX.sub("", normalized)
    normalized = _BEDROCK_VERSION_SUFFIX.sub("", normalized)
    return normalized


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Per-MTok USD pricing for one model, with cache-tier defaults."""

    input_per_mtok_usd: float
    output_per_mtok_usd: float
    cached_input_per_mtok_usd: float | None = None
    cache_write_per_mtok_usd: float | None = None
    source: str = "builtin"

    def cost_usd(self, usage: TokenUsage) -> float:
        """Compute the USD cost of `usage` at this price."""
        cached_price = (
            self.cached_input_per_mtok_usd
            if self.cached_input_per_mtok_usd is not None
            else self.input_per_mtok_usd * 0.1
        )
        write_price = (
            self.cache_write_per_mtok_usd
            if self.cache_write_per_mtok_usd is not None
            else self.input_per_mtok_usd * 1.25
        )
        return (
            usage.input_tokens / 1_000_000 * self.input_per_mtok_usd
            + usage.cached_input_tokens / 1_000_000 * cached_price
            + usage.cache_write_tokens / 1_000_000 * write_price
            + usage.billable_output / 1_000_000 * self.output_per_mtok_usd
        )


@lru_cache(maxsize=1)
def _load_builtin_raw() -> dict[str, Any]:
    data = (
        resources.files("dark_factory.pricing").joinpath("prices.yml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(data) or {}


@dataclass
class PriceBook:
    """A layered set of per-model prices with an unknown-model policy."""

    _entries: dict[str, ModelPrice] = field(default_factory=dict)
    _patterns: dict[str, ModelPrice] = field(default_factory=dict)
    on_unknown_model: UnknownModelPolicy = "error"

    @classmethod
    def builtin(cls, *, on_unknown_model: UnknownModelPolicy = "error") -> PriceBook:
        """Load the shipped `prices.yml` price table."""
        raw = _load_builtin_raw()
        entries = {
            key: ModelPrice(**value, source="builtin")
            for key, value in raw.get("models", {}).items()
        }
        patterns = {
            key: ModelPrice(**value, source="builtin")
            for key, value in raw.get("patterns", {}).items()
        }
        return cls(_entries=entries, _patterns=patterns, on_unknown_model=on_unknown_model)

    def with_overrides(self, overrides: Mapping[str, Mapping[str, float]]) -> PriceBook:
        """Return a new PriceBook with config-supplied prices layered on top."""
        entries = dict(self._entries)
        for key, value in overrides.items():
            entries[key] = ModelPrice(**value, source="config")
        return PriceBook(
            _entries=entries, _patterns=dict(self._patterns), on_unknown_model=self.on_unknown_model
        )

    def price_for(self, provider: str, model: str) -> ModelPrice:
        """Resolve a model's price via exact, normalized, glob, then policy fallback.

        Raises:
            ModelNotPricedError: If nothing matches and `on_unknown_model` is
                "error" and the provider isn't free-by-default.
        """
        key = f"{provider}:{model}"
        if key in self._entries:
            return self._entries[key]

        normalized_key = f"{provider}:{normalize_model_id(model)}"
        if normalized_key in self._entries:
            return self._entries[normalized_key]

        best_match: tuple[int, ModelPrice] | None = None
        for pattern_key, price in self._patterns.items():
            pattern_provider, _, pattern_glob = pattern_key.partition(":")
            if pattern_provider != provider:
                continue
            if fnmatch.fnmatch(model, pattern_glob):
                specificity = len(pattern_glob)
                if best_match is None or specificity > best_match[0]:
                    best_match = (specificity, price)
        if best_match is not None:
            return best_match[1]

        if provider in FREE_BY_DEFAULT_PROVIDERS:
            return ModelPrice(
                input_per_mtok_usd=0.0, output_per_mtok_usd=0.0, source="free-default"
            )

        if self.on_unknown_model == "zero":
            return ModelPrice(
                input_per_mtok_usd=0.0, output_per_mtok_usd=0.0, source="unknown-zero"
            )
        if self.on_unknown_model == "warn_fallback":
            # Conservative synthetic price rather than $0, so a ceiling still bites.
            return ModelPrice(
                input_per_mtok_usd=5.0, output_per_mtok_usd=25.0, source="unknown-fallback"
            )

        raise ModelNotPricedError(provider, model)

    def cost_usd(self, provider: str, model: str, usage: TokenUsage) -> float:
        """Resolve the price for `provider`/`model` and cost out `usage`."""
        return self.price_for(provider, model).cost_usd(usage)

    def explain(self, provider: str, model: str) -> str:
        """Return a human-readable summary of the resolved price and its source."""
        price = self.price_for(provider, model)
        return (
            f"{provider}:{model} -> ${price.input_per_mtok_usd}/MTok in, "
            f"${price.output_per_mtok_usd}/MTok out (source: {price.source})"
        )
