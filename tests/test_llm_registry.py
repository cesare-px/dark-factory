import sys

import pytest

from dark_factory.llm.errors import UnknownProviderError
from dark_factory.llm.registry import known_providers, register, resolve


def test_import_dark_factory_touches_no_provider_sdk():
    # These packages are never installed in this environment; if
    # dark_factory imported them eagerly, this import would already have
    # failed before this test even ran.
    for module_name in ("anthropic", "openai", "boto3", "google.genai"):
        assert module_name not in sys.modules


def test_known_providers_lists_all_builtins():
    providers = known_providers()
    for expected in ("mock", "anthropic", "openai", "openai_compatible"):
        assert expected in providers


def test_resolve_mock_does_not_import_real_sdks():
    resolve("mock")
    assert "anthropic" not in sys.modules
    assert "openai" not in sys.modules


def test_unknown_provider_raises_with_suggestion():
    with pytest.raises(UnknownProviderError) as excinfo:
        resolve("antropic")
    assert "anthropic" in str(excinfo.value)


def test_register_overrides_builtin_lookup():
    from dark_factory.llm.registry import _REGISTERED

    def factory(spec):
        return "sentinel-client"

    register("mock", factory)
    try:
        assert resolve("mock") is factory
        # The registered factory ignores its spec argument entirely, so a
        # real ResolvedLLM isn't needed to prove dispatch works.
        assert resolve("mock")(None) == "sentinel-client"  # type: ignore[arg-type]
    finally:
        _REGISTERED.pop("mock", None)
