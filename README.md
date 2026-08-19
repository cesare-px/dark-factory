# dark-factory

Central orchestrator core (Repo A) for the Dark Software Factory: an
autonomous, multi-agent software engineering pipeline driven by GitHub
Issues. See `dark-software-factory-blueprint.md` for the full architecture.

Provider-agnostic (Anthropic, OpenAI, or any OpenAI-compatible endpoint --
Kimi, GLM, Qwen, Ollama, vLLM, ...) and configured entirely through one YAML
file per downstream repo, with sane defaults if you delete it.

## Quickstart for a downstream repo

1. Copy `templates/project-repo-template/.github/workflows/caller.yml` into
   your repo at the same path.
2. Copy `templates/project-repo-template/.dark-factory.yml` into your repo
   root and edit `llm.provider`/`llm.model` (or delete it entirely for
   defaults).
3. Add repo secrets: `DARK_FACTORY_APP_ID`, `DARK_FACTORY_APP_PRIVATE_KEY`
   (the GitHub App installed on your repo), and `LLM_API_KEY` (skip this one
   if you're pointing at a free local endpoint).
4. Run `dark-factory doctor` from your repo root to confirm everything
   resolves before the factory ever spends money.

## Layout

```
src/dark_factory/
  intake/       webhook -> unified FactoryTicket schema, incl. HMAC
                signature verification and prompt-injection sanitization
  llm/          provider-agnostic LLMClient protocol, lazy provider
                registry (mock/anthropic/openai/openai_compatible), and the
                MeteredLLMClient that charges every call to the budget
  pricing/      per-model USD pricing (prices.yml), decoupled from any
                provider SDK or config format
  config/       FactoryConfig dataclasses + loader (.dark-factory.yml ->
                env vars -> explicit overrides, with provenance tracking)
  guardrails/   TokenBudgetTracker ($ ceiling) and IterationLoopTracker
                (build-test-fix retry cap)
  harness.py    the configured test command (SubprocessHarness) plus a
                ScriptedHarness for offline tests
  context.py    repo file scanning for the Planner (git-grep / glob)
  naming.py     branch/PR naming from templates, with git-ref validation
  labels.py     LabelKey enum -- agents never hardcode label strings
  agents/       Validator / Planner / Developer / Reviewer agents and the
                DarkFactoryPipeline state machine that runs them
  cli.py        `dark-factory run|doctor|config|prices|init`

tests/          pytest coverage for every layer above, all offline

templates/project-repo-template/
  .github/workflows/caller.yml           downstream repo's trigger workflow
  .github/ISSUE_TEMPLATE/dark-factory-task.md
  .dark-factory.yml                      annotated example config

.github/workflows/orchestrate.yml        the reusable workflow caller.yml calls
```

## Configuration reference

Every key is optional; unset ones fall back to the default shown.

| Key | Default | Notes |
|---|---|---|
| `llm.provider` | `mock` | `anthropic`, `openai`, `openai_compatible`, or `mock` |
| `llm.model` | `mock-default` | provider-specific model id |
| `llm.api_key_env` | `LLM_API_KEY` | name of the env var holding the key -- never the key itself |
| `llm.preset` | none | `moonshot`, `zhipu`, `dashscope`, `ollama` (only for `openai_compatible`) |
| `llm.base_url` | none | any OpenAI-compatible endpoint; overrides `preset` |
| `models.<name>` | -- | named model presets, referenced by `agents.<agent>.use` |
| `agents.<agent>.use` | none | switch that agent to a named preset from `models` |
| `agents.<agent>.llm` | none | sparse per-agent override (only the keys you set) |
| `agents.developer.max_iterations` | `loop.max_iterations` | per-agent override of the retry cap |
| `budget.max_usd` | `3.00` | hard USD ceiling for one ticket's whole pipeline run |
| `budget.max_usd_per_agent` | `{}` | optional sub-ceiling per agent name |
| `loop.max_iterations` | `5` | build-test-fix retry cap |
| `test_harness.command` | `./factory-test.sh` | the command the Developer agent runs to check its work |
| `intake.min_description_chars` | `30` | Validator's structural minimum |
| `labels.*` | `spec-validated`, `needs-specification`, `factory-blocked`, `factory-shipped` | rename any label string |
| `pricing.on_unknown_model` | `error` | `error`, `warn_fallback`, or `zero` for a model missing from the built-in price table |

Run `dark-factory config --explain` to see every effective value and which
layer (defaults / file / env / override) set it.

## Development

```bash
make install     # pip install -e ".[dev]"
make check       # lint + format-check + typecheck + test -- exactly what CI runs
make help        # list every shortcut
```

Or run the tools directly:

```bash
ruff check .          # lint (PEP 8, import order, docstrings, bugbear, ...)
ruff format .         # format (100-char lines)
mypy                  # strict type checking (src/); a lighter standard for tests/
pytest                # test suite
pytest --cov=dark_factory --cov-report=term-missing   # or: make test-cov
```

`make check` (equivalently `make ci`) runs in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push and PR.
Docstrings follow the Google convention and are required on public modules,
classes, and functions in `src/` (`tests/` is exempt -- test names are
self-documenting). See `[tool.ruff]` and `[tool.mypy]` in
[pyproject.toml](pyproject.toml) for the exact rule set.

## Notes

- All untrusted text (issue titles/bodies, and test-harness stdout/stderr fed
  back into the Developer agent) passes through `dark_factory.intake.sanitize`
  before it reaches a prompt. Flags are attached to the `FactoryTicket`, not
  silently dropped, so suspicious tickets remain auditable end to end.
- `provider: mock` (and any offline `ScriptedHarness`/`ScriptedScanner`) is
  what the test suite runs against -- zero network calls, zero cost,
  deterministic. `dark-factory run --dry-run` forces the same mock provider
  for a smoke test against a real event payload.
