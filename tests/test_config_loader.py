from pathlib import Path

import pytest

from dark_factory.config import ConfigError, load_config
from dark_factory.config.env import parse_env_overrides
from dark_factory.config.loader import find_config_file


def test_zero_config_gives_working_defaults():
    cfg = load_config(overrides={})
    assert cfg.llm.provider == "mock"
    assert cfg.budget.max_usd == 3.00
    assert cfg.loop.max_iterations == 5
    assert "pytest" in cfg.tool_use.safe_command_prefixes
    # Generic file-reading utilities must never be always-on defaults --
    # they have no notion of "the repo" and would bypass every other
    # tool's path containment. The dedicated read_file/list_files/grep
    # tools cover this need safely; run_command must not re-open it.
    assert "cat" not in cfg.tool_use.safe_command_prefixes
    assert "ls" not in cfg.tool_use.safe_command_prefixes
    assert "grep" not in cfg.tool_use.safe_command_prefixes


def test_tool_use_command_families_load_from_config():
    cfg = load_config(
        overrides={
            "tool_use": {
                "max_iterations": 12,
                "command_families": {"Package installation": ["pip install", "npm install"]},
            }
        }
    )
    assert cfg.tool_use.max_iterations == 12
    assert cfg.tool_use.command_families["Package installation"] == ["pip install", "npm install"]


def test_file_beats_defaults_env_beats_file_overrides_beat_env(tmp_path: Path):
    config_path = tmp_path / ".dark-factory.yml"
    config_path.write_text("llm:\n  provider: anthropic\n  model: from-file\n", encoding="utf-8")

    cfg = load_config(
        root=tmp_path,
        env={"DARK_FACTORY_MODEL": "from-env"},
        overrides={"llm": {"model": "from-override"}},
    )
    assert cfg.llm.provider == "anthropic"  # from file, untouched by higher layers
    assert cfg.llm.model == "from-override"  # override wins over env and file


def test_find_config_file_discovery_order(tmp_path: Path):
    assert find_config_file(tmp_path) is None
    (tmp_path / ".dark-factory.yml").write_text("llm:\n  provider: mock\n", encoding="utf-8")
    assert find_config_file(tmp_path) == tmp_path / ".dark-factory.yml"


def test_unknown_top_level_key_reports_suggestion():
    with pytest.raises(ConfigError) as excinfo:
        load_config(overrides={"budgt": {"max_usd": 1}})
    assert "budgt" in str(excinfo.value)


def test_unknown_nested_key_reports_suggestion_and_choices():
    with pytest.raises(ConfigError) as excinfo:
        load_config(overrides={"llm": {"provdier": "anthropic"}})
    message = str(excinfo.value)
    assert "provdier" in message
    assert "provider" in message


def test_secret_lookalike_value_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        load_config(overrides={"llm": {"api_key_env": "sk-ant-api03-thisisnotreal1234567890"}})
    assert "secret" in str(excinfo.value).lower()


def test_bad_numeric_string_reports_actionable_error():
    with pytest.raises(ConfigError) as excinfo:
        load_config(overrides={"budget": {"max_usd": "not-a-number"}})
    assert "expected a number" in str(excinfo.value)


def test_quoted_number_is_coerced_not_rejected():
    cfg = load_config(overrides={"budget": {"max_usd": "3.00"}})
    assert cfg.budget.max_usd == 3.0
    assert isinstance(cfg.budget.max_usd, float)


def test_agent_llm_override_only_replaces_specified_keys():
    cfg = load_config(
        overrides={
            "llm": {"provider": "anthropic", "model": "claude-sonnet-5", "max_output_tokens": 4096},
            "agents": {"developer": {"llm": {"model": "claude-opus-5"}}},
        }
    )
    resolved = cfg.resolve_llm("developer")
    assert resolved.model == "claude-opus-5"
    assert resolved.provider == "anthropic"  # not clobbered by LLMSettings() defaults
    assert resolved.max_output_tokens == 4096


def test_agent_use_selects_named_model_preset():
    cfg = load_config(
        overrides={
            "models": {"cheap": {"provider": "anthropic", "model": "claude-haiku-4-5"}},
            "agents": {"validator": {"use": "cheap"}},
        }
    )
    resolved = cfg.resolve_llm("validator")
    assert resolved.provider == "anthropic"
    assert resolved.model == "claude-haiku-4-5"


def test_agent_use_unknown_preset_raises():
    cfg = load_config(overrides={"agents": {"validator": {"use": "does-not-exist"}}})
    with pytest.raises(KeyError):
        cfg.resolve_llm("validator")


def test_max_iterations_for_falls_back_to_loop_default():
    cfg = load_config(overrides={"loop": {"max_iterations": 7}})
    assert cfg.max_iterations_for("developer") == 7
    cfg2 = load_config(
        overrides={"loop": {"max_iterations": 7}, "agents": {"developer": {"max_iterations": 2}}}
    )
    assert cfg2.max_iterations_for("developer") == 2


def test_env_nesting_separator():
    parsed = parse_env_overrides({"DARK_FACTORY_BUDGET__MAX_USD": "1.5", "IRRELEVANT": "x"})
    assert parsed == {"budget": {"max_usd": 1.5}}


def test_env_short_aliases():
    parsed = parse_env_overrides({"DARK_FACTORY_MODEL": "gpt-5", "DARK_FACTORY_PROVIDER": "openai"})
    assert parsed == {"llm": {"model": "gpt-5", "provider": "openai"}}
